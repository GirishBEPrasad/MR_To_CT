import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. Load & Preprocess Data
# ---------------------------
class MRICTDataset(Dataset):
    def __init__(self, mri_dir, ct_dir, img_size=(128, 128)):
        self.mri_files = sorted(os.listdir(mri_dir))
        self.ct_files = sorted(os.listdir(ct_dir))
        self.mri_dir = mri_dir
        self.ct_dir = ct_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])

        # Load images as grayscale
        mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)

        # Resize & Normalize
        mri = cv2.resize(mri, self.img_size) / 255.0
        ct = cv2.resize(ct, self.img_size) / 255.0

        # Convert to PyTorch tensors
        mri = torch.tensor(mri, dtype=torch.float32).unsqueeze(0)
        ct = torch.tensor(ct, dtype=torch.float32).unsqueeze(0)

        return mri, ct

# Dataset Paths (Update to your data locations)
mri_dir = "dataset/MRI"
ct_dir = "dataset/CT"

# Load dataset
dataset = MRICTDataset(mri_dir, ct_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Loaded {len(dataset)} MRI-CT image pairs")

# ---------------------------
# 2. Define U-Net Model
# ---------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding Path (Decoder)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Final Output Layer
        self.final = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)

        # MaxPooling and Upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final(d1)

# Instantiate model
model = UNet()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
print(model)

# ---------------------------
# 3. Train the Model
# ---------------------------
def train_model(model, dataloader, epochs=10, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    criterion = nn.MSELoss()  # Mean Squared Error loss for pixel-wise similarity
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for mri, ct in dataloader:
            mri, ct = mri.to(device), ct.to(device)

            optimizer.zero_grad()
            pred_ct = model(mri)

            loss = criterion(pred_ct, ct)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Train the model
train_model(model, dataloader, epochs=10, lr=0.0001)

# ---------------------------
# 4. Test on a New MRI Scan
# ---------------------------
def predict_and_display(model, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Load test MRI
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Generate Pseudo-CT
    with torch.no_grad():
        pred_ct = model(img_tensor).cpu().squeeze(0).squeeze(0).numpy()

    # Display
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized, cmap="gray")
    plt.title("Input MRI")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_ct, cmap="jet", alpha=0.7)
    plt.title("Generated CT")

    plt.show()

# Test on a new MRI image
test_mri = "dataset/test_mri.jpg"
predict_and_display(model, test_mri)
