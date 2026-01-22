import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import *

class modelchm(nn.Module):
    """Pytorch segmentation U-Net adapted for regression."""

    def __init__(
        self,
        img_size,
        num_channels,
        num_classes, 
        pretrained_encoder=True,
        pretrained_encoder_path=None,
        chkpt_path=None,
    ):
        super().__init__()
        if pretrained_encoder and pretrained_encoder_path is None:
            encoder_weights = "imagenet"
        else:
            encoder_weights = None

        self.seg_model = smp.create_model(
            arch="unet",
            encoder_name="mit_b5",
            classes=num_classes, 
            in_channels=num_channels,
            encoder_weights=encoder_weights,
        )

        if pretrained_encoder_path:
            self.seg_model.encoder.load_state_dict(
                torch.load(pretrained_encoder_path)
            )

        set_first_layer(self.seg_model.encoder, num_channels)

        if chkpt_path:
            strict = True
            chkpt = torch.load(chkpt_path)["state_dict"]
            chkpt = {k.replace("model.seg_model.", ""): v for k, v in chkpt.items()}
            
            if chkpt["segmentation_head.0.weight"].shape[0] != num_classes:
                chkpt = {k: v for k, v in chkpt.items() if "segmentation_head" not in k}
                strict = False
                print("Num classes different from checkpoint, segmentation head reinitialized.")

            if chkpt["encoder.conv1.weight"].shape[1] != num_channels:
                 chkpt = {k: v for k, v in chkpt.items() if "encoder.conv1" not in k}
                 strict = False
                 print("Num channels different from checkpoint, encoder conv1 reinitialized.")

            self.seg_model.load_state_dict(chkpt, strict=strict)

    def forward(self, x, metas=None):
        output = self.seg_model(x)
        return {"out": output}
