
import os
import json
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import WasteCNN

from data_loader import (
    train_loader,
    val_loader,
    test_loader,
    class_names,
    IMAGE_SIZE
)


# -----------------------------
# AYARLAR
# -----------------------------

SEED = 42
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# -----------------------------
# DEVICE SEÇİMİ
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\nKullanılan cihaz:", device)


# -----------------------------
# MODEL
# -----------------------------

model = WasteCNN(
    num_classes=len(class_names),
    pretrained=True
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    average_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# -----------------------------
# TRAINING
# -----------------------------

best_val_accuracy = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()

print("\nEğitim başlıyor...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)

    train_accuracy = 100 * correct / total

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy * 100)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE
            },
            "models/best_model.pth"
        )

        print("En iyi model kaydedildi!")


training_time = time.time() - start_time

print("\nEğitim tamamlandı!")

print(
    f"En iyi Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Eğitim süresi: "
    f"{training_time / 60:.2f} dakika"
)


# -----------------------------
# BEST MODELİ YÜKLE
# -----------------------------

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# -----------------------------
# TEST
# -----------------------------

all_labels = []
all_predictions = []

test_start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

if device.type == "mps":
    torch.mps.synchronize()

elif device.type == "cuda":
    torch.cuda.synchronize()

total_test_time = time.time() - test_start

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nTEST SONUÇLARI")
print("-----------------------------")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report:\n")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")

plt.close()


# -----------------------------
# TRAINING GRAPHS
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig("results/egitim_grafik.png")

plt.close()


plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig("results/accuracy_grafik.png")

plt.close()


# -----------------------------
# SINGLE IMAGE INFERENCE TIME
# -----------------------------

sample_images, _ = next(iter(test_loader))

sample_image = sample_images[0:1].to(device)

with torch.no_grad():

    for _ in range(10):
        model(sample_image)

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.time()

    for _ in range(100):
        model(sample_image)

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (time.time() - inference_start) / 100

print(
    f"\nTek görüntü ortalama inference süresi: "
    f"{inference_time * 1000:.2f} ms"
)


# -----------------------------
# METRİKLERİ KAYDET
# -----------------------------

metrics = {
    "model": "ResNet18",
    "image_size": IMAGE_SIZE,
    "epochs": EPOCHS,
    "batch_size": 32,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss_function": "CrossEntropyLoss",
    "device": str(device),
    "training_time_minutes": training_time / 60,
    "best_validation_accuracy": best_val_accuracy * 100,
    "test_accuracy": test_accuracy * 100,
    "inference_time_ms": inference_time * 1000,
    "classification_report": report,
    "confusion_matrix": cm.tolist()
}

with open("results/metrics.json", "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("\nSonuçlar results/metrics.json dosyasına kaydedildi.")

print("\nTüm işlemler tamamlandı!")
