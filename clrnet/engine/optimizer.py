import torch


def _build_param_groups(net, cfg_cp):
    param_group_cfgs = cfg_cp.pop('param_groups', None)
    if not param_group_cfgs:
        return net.parameters()

    named_params = list(net.named_parameters())
    assigned = set()
    params = []

    for group_cfg in param_group_cfgs:
        group_cfg = group_cfg.copy()
        patterns = group_cfg.pop('patterns')
        lr_mult = group_cfg.pop('lr_mult', 1.0)
        decay_mult = group_cfg.pop('decay_mult', 1.0)

        group_params = []
        for name, param in named_params:
            if not param.requires_grad or name in assigned:
                continue
            if any(pattern in name for pattern in patterns):
                group_params.append(param)
                assigned.add(name)

        if not group_params:
            continue

        group = {'params': group_params}
        if 'lr' in cfg_cp:
            group['lr'] = cfg_cp['lr'] * lr_mult
        if 'weight_decay' in cfg_cp:
            group['weight_decay'] = cfg_cp['weight_decay'] * decay_mult
        group.update(group_cfg)
        params.append(group)

    default_params = [
        param for name, param in named_params
        if param.requires_grad and name not in assigned
    ]
    if default_params:
        params.append({'params': default_params})

    return params


def build_optimizer(cfg, net):
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(torch.optim, cfg_type)
    params = _build_param_groups(net, cfg_cp)
    return _optim(params, **cfg_cp)
