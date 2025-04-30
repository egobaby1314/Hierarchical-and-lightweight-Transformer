# 文件: configs/xray/retinanet_pvt-tiny_xray.py
# --------------------------------------------------
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',     # 保留 RetinaNet 头部、FPN、训练设置
    '../_base_/datasets/voc0712.py',            # **换成 VOC 基础配置**
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ----------1) 类别 ----------
classes = ('Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors',)
num_classes = len(classes)          # = 5

# ----------2) 模型 ----------
model = dict(
    type='RetinaNet',
    backbone=dict(                  # 用 PVT-Tiny 做主干
        _delete_=True,
        type='PyramidVisionTransformer',
        num_layers=[2, 2, 2, 2],    # Tiny 版
        init_cfg=dict(
            type='Pretrained',      # 如果不想用 COCO 检测权重，仅留这一行即可
            checkpoint='https://github.com/whai362/PVT/releases/'
                       'download/v2/pvt_tiny.pth')
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    bbox_head=dict(num_classes=num_classes)     # 把 80 改成 5
)

# ----------3) 数据集根目录 ----------
data_root = r'C:/Users/ZHAO XINFENG/Downloads/VOC2007_SIXRAY/'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file='ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root=''),
        metainfo=dict(classes=classes))
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file='ImageSets/Main/val.txt',
        data_prefix=dict(sub_data_root=''),
        metainfo=dict(classes=classes))
)
test_dataloader = val_dataloader
val_evaluator = dict(type='VOCMetric', metric='mAP')
test_evaluator = val_evaluator

# ----------4) 训练参数 ----------
max_epochs = 4                      # VOC0712 默认 4×3=12 epoch，可自行调
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs,
         by_epoch=True, milestones=[3], gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
)
auto_scale_lr = dict(enable=False, base_batch_size=16)


