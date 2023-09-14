import torch
from loguru import logger

import common.torch_utils as torch_utils
from common.xdict import xdict
from src.callbacks.loss.loss_transformer_mf import compute_loss
from src.callbacks.process.process_arctic import process_data
from src.callbacks.vis.visualize_arctic import visualize_all
from src.models.generic.wrapper import GenericWrapper
from src.models.transformer_mf.model import TransformerMF


class TransformerMFWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = TransformerMF(
            backbone=args.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "cdev",
            "mrrpe",
            "mpjpe.ra",
            "aae",
            "success_rate",
        ]

        self.vis_fns = [visualize_all]
        self.num_vis_train = 0
        self.num_vis_val = 1

    def set_training_flags(self):
        if not self.started_training:
            if self.args.img_feat_version:
                sd_p = f"./logs/{self.args.img_feat_version}/checkpoints/last.ckpt"
                sd = torch.load(sd_p)["state_dict"]
                missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
                torch_utils.toggle_parameters(self, True)
                logger.info(f"Loaded: {sd_p}")
                logger.info(f"Missing keys: {missing_keys}")
                logger.info(f"Unexpected keys: {unexpected_keys}")
                self.model.loaded()
        self.started_training = True

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)
