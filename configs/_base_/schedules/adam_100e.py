# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[1, 2, 2])
runner = dict(type='EpochBasedRunner', max_epochs=1)
