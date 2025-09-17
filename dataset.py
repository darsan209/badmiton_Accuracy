import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class BadmintonVideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform=None, frames_per_clip=16):
        self.annotations = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.annotations.iloc[idx, 0])  # video filename
        label = int(self.annotations.iloc[idx, 1])  # label (must be numeric)

        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample evenly spaced frames
        indices = torch.linspace(0, total_frames - 1, self.frames_per_clip).long()
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # Stack into tensor [T, C, H, W]
        video_tensor = torch.stack(frames) if len(frames) > 0 else torch.zeros(self.frames_per_clip, 3, 128, 128)

        return video_tensor, label





