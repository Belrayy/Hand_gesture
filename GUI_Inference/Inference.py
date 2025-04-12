import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import random
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore
import sys
sys.path.append("Pre_trained/Inference/temporal-shift-module")  # Add TSM repo to Python path

from models import TSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = TSM(
    num_classes=27,                  # 27 classes for Jester
    num_segments=8,                  # Matches pretrained model config
    base_model='resnet50',           # Backbone architecture
    shift_type='TSM',                # Temporal Shift Module
    non_local=False,                 # Disable non-local blocks
    pretrained='imagenet'            # Initialize with ImageNet weights
)

checkpoint = torch.load('Pre_trained/PC.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval().to(device)