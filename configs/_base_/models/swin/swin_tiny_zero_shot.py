# model settings
model = dict(
    type='Recognizer3DZeroShot',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(4,4,4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True),
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
    test_cfg = dict(average_clips='prob'))
