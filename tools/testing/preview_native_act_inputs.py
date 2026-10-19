#!/usr/bin/env python3
"""Show actual decoded, letterboxed dataset images before ACT ImageNet scaling."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train_g1_native_lerobot as baseline
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset


def main():
    torch.set_num_threads(2)
    root=baseline.OUTPUT
    cfg=PreTrainedConfig.from_pretrained(root/"run/checkpoints/001000/pretrained_model")
    cfg.device="cpu"
    data=make_dataset(TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/g1_0923_pick_apple",
        root=str(root/"dataset"),video_backend="pyav",use_imagenet_stats=True),policy=cfg))
    data.image_transforms=baseline.Letterbox()
    frames=(52,104,156)
    items=[data[f] for f in frames]
    fig,axes=plt.subplots(3,3,figsize=(10,10),constrained_layout=True)
    for row,key in enumerate(data.meta.camera_keys):
        for col,frame in enumerate(frames):
            image=items[col][key].permute(1,2,0).numpy()
            axes[row,col].imshow(image.clip(0,1))  # Display only: remove FP ulp overshoot.
            axes[row,col].set_title(f"{key.split('.')[-1]} | ep0 frame{frame}",fontsize=9)
            axes[row,col].axis("off")
    fig.suptitle("Actual dataset input: 224 letterbox, before ACT ImageNet mean/std",fontsize=12)
    output=root.parent/"model_input_preview.png"
    fig.savefig(output,dpi=130)
    plt.close(fig)
    print("MODEL_INPUT_PREVIEW_COMPLETE",output,flush=True)


if __name__=="__main__":main()
