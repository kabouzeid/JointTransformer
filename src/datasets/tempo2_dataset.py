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
        getitem = self.getitem_eval if eval else self.getitem

        center_imgname = self.imgnames[index]
        imgnames = self.imgnames_window(center_imgname)

        inputs_list = []
        targets_list = []
        meta_list = []
        for imgname in imgnames:
            inputs, targets, meta_info = getitem(imgname)
            inputs_list.append(inputs)
            targets_list.append(targets)
            meta_list.append(meta_info)

        center_idx = self.window_size // 2
        assert center_imgname == imgnames[center_idx]
        inputs, targets, meta_info = (
            inputs_list[center_idx],
            targets_list[center_idx],
            meta_list[center_idx],
        )

        inputs["img_window"] = ld_utils.stack_dl(
            ld_utils.ld2dl(inputs_list), dim=0, verbose=False
        )[
            "img"
        ]  # here we get the full window of image frames

        return inputs, targets, meta_info

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
