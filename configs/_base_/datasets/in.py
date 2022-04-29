dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(type='Scale'),
    dict(type='Pad', patch=5),
    dict(type='Sampling', ratio=0.1),
    dict(type='ExtractPatch'),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        manner='IN',
        data_prefix='G:/git_leeguandong/mmhyperspectral/data/IN/Indian_pines_corrected.mat',
        data_gt='G:/git_leeguandong/mmhyperspectral/data/IN/Indian_pines_gt.mat',
        pipeline=train_pipeline))
