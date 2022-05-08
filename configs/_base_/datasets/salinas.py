dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(type='Normalize'),
    dict(type='Pad', patch=5),
    dict(type='Sampling', ratio=0.1),
    dict(type='ExtractPatch'),
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        manner='Salinas',
        data_prefix='.../data/IN/Salinas_corrected.mat',
        data_gt='.../data/IN/Salinas_gt.mat',
        pipeline=train_pipeline))

evaluation = dict(interval=1, metric='accuracy')
