from src.callbacks.loss.loss_transformer_sh_sf import compute_loss
from src.callbacks.process.process_freihand import process_data
from src.callbacks.vis.visualize_freihand import visualize_all
from src.models.generic.wrapper_sh import GenericWrapperSH
from src.models.transformer_sh_sf.model import TransformerSHSF


class TransformerSHSFWrapper(GenericWrapperSH):
    def __init__(self, args):
        super().__init__(args)
        self.model = TransformerSHSF(
            backbone=args.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            # "cdev",
            # "mrrpe",
            # "mpjpe.ra",
            # "aae",
            # "success_rate",
        ]

        self.vis_fns = [visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)
