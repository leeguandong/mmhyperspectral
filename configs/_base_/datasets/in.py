dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(),

]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_prefix='G:/git_leeguandong/mmhyperspectralcls/data/IN',
        pipeline=train_pipeline
    ))
