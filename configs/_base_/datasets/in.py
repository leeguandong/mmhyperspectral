dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(type='Scale'),
    dict(type='Pad', patch=5),
    dict(type='Sampling', ratio=0.1),

]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        manner='IN',
        data_prefix='G:/git_leeguandong/mmhyperspectralcls/data/IN/Indian_pines_corrected.mat',
        data_gt='G:/git_leeguandong/mmhyperspectralcls/data/IN/Indian_pines_gt.mat',
        total_size=10249,
        split_ratio=0.1,
        pipeline=train_pipeline))
