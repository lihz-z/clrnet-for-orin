import argparse
import json
import math
import os
import re
import statistics
from datetime import datetime


TRAIN_LINE_RE = re.compile(
    r'epoch:\s*(?P<epoch>\d+)\s+step:\s*(?P<step>\d+)\s+lr:\s*(?P<lr>[0-9.eE+-]+)\s+'
    r'(?P<rest>.*)$')
METRIC_LINE_RE = re.compile(r'metric:\s*(?P<metric>[0-9.]+)')
KV_RE = re.compile(r'([A-Za-z0-9_]+):\s*([0-9:.eE+-]+)')
WORK_DIR_RE = re.compile(r'work_dirs\s*=\s*[\'"]([^\'"]+)[\'"]')
EPOCHS_RE = re.compile(r'^epochs\s*=\s*(\d+)\s*$', re.MULTILINE)
BATCH_SIZE_RE = re.compile(r'^batch_size\s*=\s*(\d+)\s*$', re.MULTILINE)
LR_RE = re.compile(r'optimizer\s*=\s*dict\([\s\S]*?\blr\s*=\s*([0-9.eE+-]+)',
                   re.MULTILINE)
TIMESTAMP_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')


def parse_timestamp(line):
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')


def to_number(value):
    if ':' in value:
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_train_line(line):
    match = TRAIN_LINE_RE.search(line)
    if not match:
        return None

    record = {
        'epoch_internal': int(match.group('epoch')),
        'epoch_human': int(match.group('epoch')) + 1,
        'step': int(match.group('step')),
        'lr': float(match.group('lr')),
    }
    for key, value in KV_RE.findall(match.group('rest')):
        record[key] = to_number(value)
    return record


def parse_config_block(text):
    build_idx = text.find('Build train loader...')
    head = text[:build_idx] if build_idx != -1 else text
    return {
        'work_dirs': first_group(WORK_DIR_RE, head),
        'epochs': first_group(EPOCHS_RE, head, cast=int),
        'batch_size': first_group(BATCH_SIZE_RE, head, cast=int),
        'base_lr': first_group(LR_RE, head, cast=float),
    }


def first_group(pattern, text, cast=None):
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1)
    return cast(value) if cast else value


def infer_ckpt_path(log_path, epoch_internal):
    log_dir = os.path.dirname(os.path.abspath(log_path))
    ckpt_path = os.path.join(log_dir, 'ckpt', f'{epoch_internal}.pth')
    return ckpt_path if os.path.isdir(os.path.join(log_dir, 'ckpt')) else None


def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    config = parse_config_block('\n'.join(lines))
    train_records = []
    metrics = []
    last_train_record = None
    start_time = None
    end_time = None

    for line in lines:
        ts = parse_timestamp(line)
        if ts is not None and start_time is None:
            start_time = ts
        if ts is not None:
            end_time = ts

        train_record = parse_train_line(line)
        if train_record is not None:
            train_records.append(train_record)
            last_train_record = train_record
            continue

        metric_match = METRIC_LINE_RE.search(line)
        if metric_match:
            metric_value = float(metric_match.group('metric'))
            metric_record = {
                'metric': metric_value,
                'epoch_internal': None,
                'epoch_human': None,
                'step': None,
                'lr': None,
                'train_snapshot': None,
                'ckpt_path': None,
            }
            if last_train_record is not None:
                metric_record['epoch_internal'] = last_train_record['epoch_internal']
                metric_record['epoch_human'] = last_train_record['epoch_human']
                metric_record['step'] = last_train_record['step']
                metric_record['lr'] = last_train_record['lr']
                metric_record['train_snapshot'] = last_train_record
                metric_record['ckpt_path'] = infer_ckpt_path(
                    log_path, last_train_record['epoch_internal'])
            metrics.append(metric_record)

    best_metric = max(metrics, key=lambda x: x['metric']) if metrics else None
    worst_metric = min(metrics, key=lambda x: x['metric']) if metrics else None

    metric_values = [item['metric'] for item in metrics]
    summary = {
        'log_path': os.path.abspath(log_path),
        'config': config,
        'start_time': start_time.isoformat(sep=' ') if start_time else None,
        'end_time': end_time.isoformat(sep=' ') if end_time else None,
        'duration_hours': round((end_time - start_time).total_seconds() / 3600.0,
                                4) if start_time and end_time else None,
        'train_record_count': len(train_records),
        'eval_count': len(metrics),
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'final_metric': metrics[-1] if metrics else None,
        'metric_mean': round(statistics.mean(metric_values), 6)
        if metric_values else None,
        'metric_std': round(statistics.pstdev(metric_values), 6)
        if len(metric_values) > 1 else 0.0 if metric_values else None,
        'metrics': metrics,
        'train_records': train_records,
    }
    return summary


def growth_from_first_best(metrics):
    if not metrics:
        return None
    first = metrics[0]['metric']
    best = max(item['metric'] for item in metrics)
    return best - first


def describe_stage(metrics):
    if len(metrics) < 3:
        return 'metric points too few to judge trend'

    best_idx = max(range(len(metrics)), key=lambda i: metrics[i]['metric'])
    tail = metrics[-3:]
    tail_mean = statistics.mean(item['metric'] for item in tail)
    best = metrics[best_idx]['metric']

    if best_idx == len(metrics) - 1:
        return 'best metric appears at the end, training may still improve'
    if best - tail_mean > 0.01:
        return 'best metric is clearly earlier than the tail, watch for overfitting or late decay'
    return 'tail stays close to the best metric, training looks relatively stable'


def format_metric_record(record):
    if not record:
        return 'N/A'
    ckpt = record['ckpt_path'] or f"ckpt/{record['epoch_internal']}.pth"
    return ('epoch={epoch_human} (internal={epoch_internal}), metric={metric:.6f}, '
            'step={step}, lr={lr:.6g}, pth={ckpt}').format(ckpt=ckpt, **record)


def print_single_report(summary, show_epochs):
    print('=' * 88)
    print(f"log: {summary['log_path']}")
    cfg = summary['config']
    print('config:')
    print(f"  work_dirs={cfg['work_dirs']}")
    print(f"  epochs={cfg['epochs']}, batch_size={cfg['batch_size']}, base_lr={cfg['base_lr']}")
    print('time:')
    print(f"  start={summary['start_time']}")
    print(f"  end={summary['end_time']}")
    print(f"  duration_hours={summary['duration_hours']}")
    print('metrics:')
    print(f"  eval_count={summary['eval_count']}, mean={summary['metric_mean']}, std={summary['metric_std']}")
    print(f"  best={format_metric_record(summary['best_metric'])}")
    print(f"  final={format_metric_record(summary['final_metric'])}")
    if summary['best_metric'] and summary['final_metric']:
        delta = summary['final_metric']['metric'] - summary['best_metric']['metric']
        print(f"  final_minus_best={delta:.6f}")
    print(f"  trend={describe_stage(summary['metrics'])}")

    best_snapshot = summary['best_metric']['train_snapshot'] if summary['best_metric'] else None
    if best_snapshot:
        print('best_epoch_train_snapshot:')
        keys = [
            'loss', 'cls_loss', 'reg_xytl_loss', 'seg_loss', 'iou_loss',
            'stage_0_acc', 'stage_1_acc', 'stage_2_acc', 'batch', 'data'
        ]
        for key in keys:
            if key in best_snapshot:
                print(f"  {key}={best_snapshot[key]}")

    if show_epochs and summary['metrics']:
        print('per_epoch_metric:')
        best_value = summary['best_metric']['metric']
        for item in summary['metrics']:
            flag = '  <-- best' if math.isclose(item['metric'], best_value) else ''
            print(f"  epoch {item['epoch_human']:>2}: {item['metric']:.6f}{flag}")


def print_compare_report(summaries):
    rows = []
    for summary in summaries:
        best = summary['best_metric']
        final = summary['final_metric']
        rows.append({
            'name': os.path.basename(summary['log_path']),
            'best_metric': best['metric'] if best else float('-inf'),
            'best_epoch': best['epoch_human'] if best else None,
            'final_metric': final['metric'] if final else None,
            'gain': growth_from_first_best(summary['metrics']),
            'work_dirs': summary['config']['work_dirs'],
        })
    rows.sort(key=lambda x: x['best_metric'], reverse=True)

    print('=' * 88)
    print('multi_log_comparison:')
    for idx, row in enumerate(rows, start=1):
        gain = 'N/A' if row['gain'] is None else f"{row['gain']:.6f}"
        final_metric = 'N/A' if row['final_metric'] is None else f"{row['final_metric']:.6f}"
        print(
            f"{idx:>2}. {row['name']}: best={row['best_metric']:.6f}, best_epoch={row['best_epoch']}, "
            f"final={final_metric}, gain_from_first_eval={gain}, work_dirs={row['work_dirs']}"
        )


def build_json_payload(summaries):
    compact = []
    for summary in summaries:
        compact.append({
            'log_path': summary['log_path'],
            'config': summary['config'],
            'start_time': summary['start_time'],
            'end_time': summary['end_time'],
            'duration_hours': summary['duration_hours'],
            'best_metric': summary['best_metric'],
            'final_metric': summary['final_metric'],
            'metric_mean': summary['metric_mean'],
            'metric_std': summary['metric_std'],
            'metrics': summary['metrics'],
        })
    return compact


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CLRNet training log.txt and compare multiple runs.')
    parser.add_argument('logs', nargs='+', help='Path(s) to log.txt')
    parser.add_argument('--show-epochs',
                        action='store_true',
                        help='Print per-epoch metric curve in text form')
    parser.add_argument('--json',
                        action='store_true',
                        help='Print machine-readable JSON after the text report')
    args = parser.parse_args()

    summaries = [parse_log(path) for path in args.logs]
    for summary in summaries:
        print_single_report(summary, show_epochs=args.show_epochs)
    if len(summaries) > 1:
        print_compare_report(summaries)
    if args.json:
        print('=' * 88)
        print(json.dumps(build_json_payload(summaries),
                         ensure_ascii=False,
                         indent=2))


if __name__ == '__main__':
    main()
