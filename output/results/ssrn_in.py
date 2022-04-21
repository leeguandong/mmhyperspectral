dataset_type = 'HyperSpectral'
train_pipeline = [
    dict(type='Scale'),
    dict(type='Pad', patch=5),
    dict(type='Sampling', ratio=0.1),
    dict(type='ExtractPatch')
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='HyperSpectral',
        manner='IN',
        data_prefix=
        'G:/git_leeguandong/mmhyperspectral/data/IN/Indian_pines_corrected.mat',
        data_gt=
        'G:/git_leeguandong/mmhyperspectral/data/IN/Indian_pines_gt.mat',
        pipeline=[
            dict(type='Scale'),
            dict(type='Pad', patch=5),
            dict(type='Sampling', ratio=0.1),
            dict(type='ExtractPatch')
        ]))
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 2)]
work_dir = '../output/results'
gpu_ids = range(0, 1)
