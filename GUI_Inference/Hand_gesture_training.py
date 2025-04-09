import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import numpy as np # type: ignore
from torch.cuda.amp import GradScaler, autocast   # type: ignore
import pandas as pd # type: ignore
import cv2 # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class HandDataset(Dataset):
    def __init__(self, root_dir,csv_file ,transform=None, frames_clip=30, resize=(100,100)):
        self.root_dir=root_dir
        self.annotations=pd.read_csv(csv_file)
        self.transform=transform
        self.frames_clip=frames_clip
        self.resize=resize

        if transform is None:
            self.transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        video_folder=os.path.join(self.root_dir,self.annotations.iloc[idx,0])
        label=self.annotations.iloc[idx,1]

        frames=[]
        frame_file=sorted([f for f in os.listdir(video_folder)if f.endswith('.jpg')])

        if len(frame_file) > self.frames_clip :
            indice=np.linspace(0, len (frame_file)-1, num=self.frames_clip, dtype=int)
            frame_file= [frame_file[i] for i in indice]

        for frame_file in frame_file[:self.frames_clip]:
            frame_path=os.path.join(video_folder,frame_file)
            frame=cv2.imread(frame_path)
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if self.transform:
                frame=self.transform(frame)

            frames.append(frame)

        while len(frames) < self.frames_clip: 
        


    