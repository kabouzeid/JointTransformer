import json
import os
import os.path as op
import sys
from pprint import pformat

import torch
from loguru import logger
from tqdm import tqdm

sys.path.append(".")
import common.thing as thing
import src.extraction.interface as interface
import src.factory as factory
from common.xdict import xdict
from src.parsers.parser import construct_args


def main():
    args = construct_args()

    args.experiment = None
    args.exp_key = "xxxxxxx"

    device = "cuda:0"
    wrapper = factory.fetch_model(args).to(device)
    assert args.load_ckpt != ""
    wrapper.load_state_dict(torch.load(args.load_ckpt)["state_dict"])
    logger.info(f"Loaded weights from {args.load_ckpt}")
    wrapper.eval()
    wrapper.to(device)
    # wrapper.metric_dict = []

    exp_key = op.abspath(args.load_ckpt).split("/")[-3]

    out_dir = op.join(args.load_ckpt.split("checkpoints")[0], "eval")

    logger.info(f"Hyperparameters: \n {pformat(args)}")

    assert args.extraction_mode in ["submit_pose"]

    out_dir = out_dir.replace("/eval", f"/submit")
    os.makedirs(out_dir, exist_ok=True)

    xyz_pred_list = []
    verts_pred_list = []
    val_loader = factory.fetch_dataloader(args, "val")
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = thing.thing2dev(batch, device)
            inputs, targets, meta_info = batch
            out_dict = wrapper.inference(inputs, meta_info)
            out_dict = xdict(out_dict)
            xyz_pred_list.append(out_dict["pred.mano.j3d.wrist.r"])
            verts_pred_list.append(out_dict["pred.mano.v3d.wrist.r"])

    xyz_pred_list = torch.cat(xyz_pred_list).tolist()
    verts_pred_list = torch.cat(verts_pred_list).tolist()
    logger.info("Done")

    import json

    # save to a json
    json_path = op.join(out_dir, "pred.json")
    with open(json_path, "w") as f:
        json.dump([xyz_pred_list, verts_pred_list], f)
    print(
        "Dumped %d joints and %d verts predictions to %s"
        % (len(xyz_pred_list), len(verts_pred_list), json_path)
    )

    # import shutil

    # zip_path = op.join(out_dir, "pred.zip")
    # shutil.make_archive(
    #     zip_path,
    #     "zip",
    #     root_dir=op.dirname(zip_path),
    #     base_dir=op.basename(zip_path),
    # )
    # logger.info(f"Your submission file as exported at {zip_path}.zip")


if __name__ == "__main__":
    main()
