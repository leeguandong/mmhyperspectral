dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(type='Normalize'),
    dict(type='Pad', patch=5),
    dict(type='Sampling', ratio=0.1),
    dict(type='ExtractPatch'),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    # dict(type='Collect', keys=['img', 'gt_label'])
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        manner='IN',
        data_prefix='../data/IN/Indian_pines_corrected.mat',
        data_gt='../data/IN/Indian_pines_gt.mat',
        pipeline=train_pipeline))

evaluation = dict(interval=1, metric='accuracy')
