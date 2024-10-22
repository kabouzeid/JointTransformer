DEFAULT_ARGS_EGO = {
    "run_on": "",
    "trainsplit": "train",
    "valsplit": "tinyval",
    "setup": "p2a",
    "method": "arctic",
    "log_every": 50,
    "eval_every_epoch": 5,
    "optimizer": "adam",
    "lr_scheduler": "step",
    "lr_dec_epoch": [],
    "num_epoch": 100,
    "lr": 1e-5,
    "lr_dec_factor": 10,
    "lr_decay": 0.1,
    "lr_warmup": 1e-7,
    "lr_min": 1e-7,
    "lr_warmup_steps": 0.05,
    "weight_decay": 0.01,
    "num_exp": 1,
    "exp_key": "",
    "batch_size": 64,
    "test_batch_size": 128,
    "temp_loader": False,
    "window_size": 11,
    "window_dilation": 1,
    "num_workers": 16,
    "img_feat_version": "",
    "eval_on": "",
    "acc_grad": 1,
    "load_from": "",
    "load_ckpt": "",
    "infer_ckpt": "",
    "resume_ckpt": "",
    "gpu_ids": [0],
    "agent_id": 0,
    "cluster_node": "",
    "bid": 21,
    "gpu_arch": "ampere",
    "gpu_min_mem": 20000,
    "extraction_mode": "",
    "backbone": "resnet50",
}
DEFAULT_ARGS_ALLO = {
    "run_on": "",
    "trainsplit": "train",
    "valsplit": "tinyval",
    "setup": "p1a",
    "method": "arctic",
    "log_every": 50,
    "eval_every_epoch": 1,
    "optimizer": "adam",
    "lr_scheduler": "step",
    "lr_dec_epoch": [],
    "num_epoch": 20,
    "lr": 1e-5,
    "lr_dec_factor": 10,
    "lr_decay": 0.1,
    "lr_warmup": 1e-7,
    "lr_min": 1e-7,
    "lr_warmup_steps": 0.05,
    "weight_decay": 0.01,
    "num_exp": 1,
    "exp_key": "",
    "batch_size": 64,
    "test_batch_size": 128,
    "window_size": 11,
    "window_dilation": 1,
    "num_workers": 16,
    "img_feat_version": "",
    "eval_on": "",
    "acc_grad": 1,
    "load_from": "",
    "load_ckpt": "",
    "infer_ckpt": "",
    "resume_ckpt": "",
    "gpu_ids": [0],
    "agent_id": 0,
    "cluster_node": "",
    "bid": 21,
    "gpu_arch": "ampere",
    "gpu_min_mem": 20000,
    "extraction_mode": "",
    "backbone": "resnet50",
}
