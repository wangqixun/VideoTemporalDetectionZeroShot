_base_ = [
    # '../../_base_/models/swin/swin_tiny_zero_shot.py', 
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3DZeroShot',
    backbone=dict(
        type='SwinTransformer3D',
        pretrained=None,
        patch_size=(2,4,4),   # 2-时间 4,4-空间
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    use_feature=True,
    nlp_backbone=dict(
        type='NLPModel',
        pretrained='/new_share/wangqixun/workspace/githup_project/model_super_strong/transformers/hfl/chinese-roberta-wwm-ext'
    ),
    # Text model 
    # video+text transformer
    # video_text+time transformer
    cls_head=dict(
        type='I3DHeadZeroShot',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob', max_testing_views=4))

# model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1), test_cfg=dict(max_testing_views=4))
# model=dict(
#     backbone=dict(
#         pretrained=None,
#         patch_size=(2,4,4), 
#         drop_path_rate=0.1
#     ), 
#     test_cfg=dict(max_testing_views=4)
# )

# dataset settings
dataset_type = 'VideoTextPositionDataset'
# dataset_type = 'VideoDataset'
data_root = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/thumos14'
data_root_val = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/thumos14'
ann_file_train = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/thumos14/annotations_val_for_zero_shot/ann_zero_shot_train.txt'
ann_file_val = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/thumos14/annotations_val_for_zero_shot/ann_zero_shot_train.txt'
ann_file_test = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/thumos14/annotations_val_for_zero_shot/ann_zero_shot_train.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/demo_debug'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

# resume_from='/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/work_dirs/v0_6/epoch_14.pth'
find_unused_parameters = True



