
import os
import urllib.request
import zipfile
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


# -----------------------------
# AYARLAR
# -----------------------------

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "dataset-resized"

URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


# -----------------------------
# VERİ SETİNİ İNDİR
# -----------------------------

if not os.path.exists(DATA_DIR):

    print("TrashNet veri seti indiriliyor...")

    urllib.request.urlretrieve(URL, "trashnet.zip")

    with zipfile.ZipFile("trashnet.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    print("Veri seti indirildi!")

else:
    print("Veri seti zaten mevcut.")


# -----------------------------
# VERİ SETİNİ YÜKLE
# -----------------------------

dataset = ImageFolder(DATA_DIR)

class_names = dataset.classes

print("\nSınıflar:", class_names)
print("Toplam görüntü:", len(dataset))


# -----------------------------
# TRAIN / VALIDATION / TEST
# -----------------------------

indices = np.arange(len(dataset))
labels = np.array(dataset.targets)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=labels[temp_indices],
    random_state=SEED
)

print("\nVeri seti ayrımı:")
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))


# -----------------------------
# NORMALIZATION
# -----------------------------

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# -----------------------------
# DATA AUGMENTATION
# -----------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.7, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    normalize
])


# Validation ve test için veri artırma yapılmaz.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])


# -----------------------------
# CUSTOM DATASET
# -----------------------------

class TrashNetDataset(Dataset):

    def __init__(self, indices, transform):
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        image, label = dataset[int(self.indices[index])]

        image = image.convert("RGB")
        image = self.transform(image)

        return image, label


# -----------------------------
# DATASET NESNELERİ
# -----------------------------

train_dataset = TrashNetDataset(
    train_indices,
    train_transform
)

val_dataset = TrashNetDataset(
    val_indices,
    eval_transform
)

test_dataset = TrashNetDataset(
    test_indices,
    eval_transform
)


# -----------------------------
# DATALOADER
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nDataLoader işlemleri tamamlandı!")
