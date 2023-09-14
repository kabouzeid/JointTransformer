import os.path as op

import numpy as np
import torch

import common.ld_utils as ld_utils
from src.datasets.arctic_dataset import ArcticDataset


class Tempo2Dataset(ArcticDataset):
    def __init__(self, args, split, seq=None):
        super().__init__(args, split, seq)

        self.aug_data = False  # do not apply augmentations
        self.window_size = args.window_size
        self.window_dilation = args.window_dilation

    def __getitem__(self, index):
        return self.getitem_tempo(index, eval=False)

    def getitem_tempo(self, index, eval):
        imgnames = self.imgnames_window(self.imgnames[index])

        targets_list = []
        meta_list = []
        inputs_list = []
        getitem = self.getitem_eval if eval else self.getitem
        for imgname in imgnames:
            inputs, targets, meta_info = getitem(imgname)
            inputs_list.append(inputs)
            targets_list.append(targets)
            meta_list.append(meta_info)

        inputs_list = ld_utils.stack_dl(
            ld_utils.ld2dl(inputs_list), dim=0, verbose=False
        )
        inputs = {"img": inputs_list["img"]}

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False)

        if not eval:
            targets_list["is_valid"] = torch.FloatTensor(
                np.array(targets_list["is_valid"])
            )
            targets_list["left_valid"] = torch.FloatTensor(
                np.array(targets_list["left_valid"])
            )
            targets_list["right_valid"] = torch.FloatTensor(
                np.array(targets_list["right_valid"])
            )
            targets_list["joints_valid_r"] = torch.FloatTensor(
                np.array(targets_list["joints_valid_r"])
            )
            targets_list["joints_valid_l"] = torch.FloatTensor(
                np.array(targets_list["joints_valid_l"])
            )
        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))
        return inputs, targets_list, meta_list

    def imgnames_window(self, imgname):
        img_idx = int(op.basename(imgname).split(".")[0])
        ind = (
            (np.arange(self.window_size) - (self.window_size - 1) / 2)
            * self.window_dilation
            + img_idx
        ).astype(np.int64)
        num_frames = self.data["/".join(imgname.split("/")[-4:-2])]["params"][
            "rot_r"
        ].shape[0]
        ind = np.clip(
            ind, 10, num_frames - 10 - 1
        )  # skip first and last 10 frames as they are not useful
        return [op.join(op.dirname(imgname), "%05d.jpg" % (idx)) for idx in ind]
