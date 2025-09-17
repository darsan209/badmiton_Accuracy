import os
import cv2
import torch
import argparse
import pandas as pd
from torchvision import transforms, models
from dataset import BadmintonVideoDataset

# -----------------------------
# Load and preprocess one video
# -----------------------------
def load_video(video_path, transform, frames_per_clip=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = torch.linspace(0, total_frames - 1, frames_per_clip).long()

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transform:
            frame = transform(frame)
        frames.append(frame)
    cap.release()

    if len(frames) > 0:
        video_tensor = torch.stack(frames)   # [T, C, H, W]
    else:
        video_tensor = torch.zeros(frames_per_clip, 3, 128, 128)

    return video_tensor.unsqueeze(0)  # add batch dim


# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate(model_path, video_path=None, csv_file=None, video_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms (same as train.py)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load model
    model = models.video.r3d_18(pretrained=False)
    # adjust fc layer depending on your dataset
    num_classes = 2   # ⚠️ CHANGE this to match your labels.csv
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # -----------------------------
    # Case 1: Predict single video
    # -----------------------------
    if video_path:
        video_tensor = load_video(video_path, transform).to(device)  # [1, T, C, H, W]
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

        with torch.no_grad():
            outputs = model(video_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        print(f"Prediction for {video_path}: class {pred_class}, probs={probs.cpu().numpy()}")

    # -----------------------------
    # Case 2: Evaluate whole dataset
    # -----------------------------
    elif csv_file and video_dir:
        dataset = BadmintonVideoDataset(csv_file, video_dir, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        correct, total = 0, 0
        with torch.no_grad():
            for videos, labels in loader:
                videos, labels = videos.to(device), labels.to(device)
                videos = videos.permute(0, 2, 1, 3, 4)
                outputs = model(videos)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Dataset evaluation: {correct}/{total} correct, Accuracy = {100*correct/total:.2f}%")

    else:
        print("⚠️ Please provide either --video for single video or --csv + --video_dir for dataset evaluation.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--video", help="Path to a single video to evaluate")
    parser.add_argument("--csv", help="Path to labels.csv for dataset evaluation")
    parser.add_argument("--video_dir", help="Folder containing videos for dataset evaluation")
    args = parser.parse_args()

    evaluate(args.model, args.video, args.csv, args.video_dir)
