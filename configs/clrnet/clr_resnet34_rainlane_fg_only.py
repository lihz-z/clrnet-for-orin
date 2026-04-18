net = dict(type='Detector')

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/r34_rainlane_fg_only"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False,
            lrf_block=dict(
                type='LRFBlock',
                channels=64,
                freq_gate_threshold=0.22,
                direction_bins=4,
                res_scale_init=0.08,
                enable_freq=True,
                enable_dir=False,
                apply_levels=[1,2]))

test_parameters = dict(conf_threshold=0.35, nms_thres=50, nms_topk=max_lanes)

epochs = 25
batch_size = 24

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    param_groups=[
        dict(patterns=['backbone'], lr_mult=0.5),
        dict(patterns=['neck.lrf_block', 'heads'], lr_mult=1.5),
    ])
auto_calc_total_iter = True
scheduler = dict(type='CosineAnnealingLR', T_max='auto')

eval_ep = 1
save_ep = 1

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(type='RainRobustAug',
         p=0.85,
         reflection_p=0.35,
         occlusion_p=0.3,
         contrast_p=0.6),
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.8, 1.2), add=(-20, 20)),
                 p=0.7),
            dict(name='LinearContrast',
                 parameters=dict(alpha=(0.75, 1.35)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-12, 12)),
                 p=0.6),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 7))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5))),
                     dict(name='GaussianBlur', parameters=dict(sigma=(0.0,
                                                                      1.8)))
                 ],
                 p=0.3),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.08, 0.08),
                                                        y=(-0.05, 0.05)),
                                 rotate=(-8, 8),
                                 scale=(0.85, 1.15)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/Rainylane_data_augmented'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 200
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
