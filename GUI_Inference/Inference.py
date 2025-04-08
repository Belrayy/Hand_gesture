import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import random
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = torch.jit.load('hand_gesture.pt').to(device)
model.eval()