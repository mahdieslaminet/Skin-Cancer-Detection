import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm

DATA_DIR = "dataset"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
META_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# ---- Load metadata
df = pd.read_csv(META_PATH)
LABELS = df["dx"].unique()
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# ---- Custom Dataset Class
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_name = row["image_id"] + ".jpg"

        img_path = os.path.join(IMG_DIR_1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR_2, img_name)

        image = Image.open(img_path).convert("RGB")
        label = LABEL2IDX[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label

# ---- Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Train/Val Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'])
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['dx'])

train_ds = SkinDataset(train_df, transform_train)
val_ds = SkinDataset(val_df, transform_test)
test_ds = SkinDataset(test_df, transform_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

# ---- Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(LABELS))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- Training Loop
EPOCHS = 5
for e in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {e+1}/{EPOCHS} | Loss = {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "skin_cancer_model.pth")
print("Model saved!")
