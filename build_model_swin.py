# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from swin_transformer import SwinTransformer
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np


def build_model(model_type):
    #odel_type = config.MODEL.TYPE
    if model_type == 'swin':
        # model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
        #                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                         in_chans=config.MODEL.SWIN.IN_CHANS,
        #                         num_classes=config.MODEL.NUM_CLASSES,
        #                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                         depths=config.MODEL.SWIN.DEPTHS,
        #                         num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                         qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         ape=config.MODEL.SWIN.APE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        #                         num_mlp_heads=config.NIH.num_mlp_heads)
        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=14,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                num_mlp_heads=3)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

if __name__ == '__main__':
    model = build_model('swin')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    #duumy_input = torch.randn(6, 3, 224, 224)
    image_path = r"C:\Users\pc\PycharmProjects\ChestXray_Classification\ChestXray_Classification\3_images_to_test\normal_256\00960.jpg"
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to Tensor and normalize to [0, 1]
    ])

    # Apply transformation
    image_tensor = transform(image)  # Shape: [3, 224, 224]

    # Add batch dimension
    bs = 1  # Change this according to your batch size
    image_batch = image_tensor.unsqueeze(0).repeat(bs, 1, 1, 1)  # Shape: [bs, 3, 224, 224]
    output = model.forward_features(image_batch)
    #output = nn.Linear(768, 1024)(output)
    print(output.shape)
    print(output)