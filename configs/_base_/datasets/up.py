dataset_type = 'HyperSpectral'

train_pipeline = [
    dict(),

]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        manner='IN',
        data_prefix='G:/git_leeguandong/mmhyperspectralcls/data/UP/PaviaU.mat',
        data_gt='G:/git_leeguandong/mmhyperspectralcls/data/UP/PaviaU_gt.mat',
        total_size=42776,
        split_ratio=0.1,
        pipeline=train_pipeline))
