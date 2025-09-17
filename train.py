import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dataset import BadmintonVideoDataset

# Paths
train_csv = "data/train/labels.csv"
train_dir = "data/train"
val_csv = "data/val/labels.csv"
val_dir = "data/val"

# Data transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Datasets & loaders
train_dataset = BadmintonVideoDataset(train_csv, train_dir, transform=transform)
val_dataset = BadmintonVideoDataset(val_csv, val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Model: pretrained 3D ResNet
model = models.video.r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.annotations['label'].unique()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for 3D CNN

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            videos = videos.permute(0, 2, 1, 3, 4)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), "outputs/models/badminton_model.pth")
