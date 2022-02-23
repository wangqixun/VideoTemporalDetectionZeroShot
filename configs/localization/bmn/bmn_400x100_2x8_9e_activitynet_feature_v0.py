_base_ = [
    '../../_base_/default_runtime.py'
]
model = dict(
    type='TBMN',
    temporal_dim=100,
    boundary_ratio=0.5,
    num_samples=32,
    num_samples_per_bin=3,
    feat_dim=400,
    soft_nms_alpha=0.4,
    soft_nms_low_threshold=0.5,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100,
    language_encoder=dict(
        pretrained_language_encoder='/new_share/wangqixun/workspace/githup_project/model_super_strong/transformers/roberta-base',
        freeze=True, 
        layers=12, 
        final_channel_number=768,
    ),
    cross_feature_decoder=dict(
        pretrained_cross_feature_decoder='/new_share/wangqixun/workspace/githup_project/model_super_strong/transformers/roberta-base',
        freeze=False, 
        layers=6, 
        final_channel_number=768,
        replace_dict=[['roberta.', '']],
    ),
)

# TODO : FasterBMNActivityNetDataset, FasterBMNLoadLocalizationFeature, FasterBMNGenerateLocalizationLabels
# dataset settings
dataset_type = 'FasterBMNActivityNetDataset'
data_root = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
data_root_val = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
ann_file_train = 'data/ActivityNet/anet_anno_train.json'
ann_file_val = 'data/ActivityNet/anet_anno_val.json'
ann_file_test = 'data/ActivityNet/anet_anno_val.json'

test_pipeline = [
    dict(type='FasterBMNLoadLocalizationFeature'),
    dict(type='FasterBMNGenerateLocalizationLabels', mode='val'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations', 'text',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
]
train_pipeline = [
    dict(type='FasterBMNLoadLocalizationFeature'),
    dict(type='FasterBMNGenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name', 'text']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))
evaluation = dict(interval=1, metrics=['AR@AN'])

# optimizer
optimizer = dict(
    type='AdamW', lr=0.00001, weight_decay=0.000001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=63)
total_epochs = 81

# runtime settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/bmn_T_v1/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')

find_unused_parameters = True