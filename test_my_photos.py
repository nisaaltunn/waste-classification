import os
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torchvision import transforms

from model import WasteCNN


# -----------------------------
# GÖRSEL AYARLARI (boyutu buradan değiştirebilirsin)
# -----------------------------

MAX_COLUMNS = 5          # yan yana en fazla kaç fotoğraf
CELL_WIDTH = 3.2         # her fotoğraf hücresinin genişliği (inç)
CELL_HEIGHT = 4.0        # her fotoğraf hücresinin yüksekliği (inç)
THUMB_SIZE = 400         # ekranda gösterilecek fotoğrafın en büyük kenarı (piksel)
PAGE_SIZE = 10           # her sayfada kaç fotoğraf gösterilsin (5 sütun x 2 satır ekrana sığar)
SHOW_WINDOW = True       # False yapılırsa pencere açılmaz, sadece PNG dosyaları kaydedilir


# -----------------------------
# DEVICE
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# -----------------------------
# MODELİ YÜKLE
# -----------------------------

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=device,
    weights_only=False
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = WasteCNN(
    num_classes=len(class_names),
    pretrained=False
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------------
# GÖRÜNTÜ DÖNÜŞÜMÜ
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# FOTOĞRAFLARI BUL
# -----------------------------

photo_dir = Path("my_photos")

extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = sorted([
    path
    for path in photo_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
])

if not image_paths:
    raise FileNotFoundError(
        "my_photos klasöründe fotoğraf bulunamadı!"
    )


# -----------------------------
# TAHMİNLERİ YAP
# -----------------------------

results = []

correct = 0
total = 0

with torch.no_grad():

    for image_path in image_paths:

        true_class = image_path.parent.name

        if true_class not in class_names:
            print(f"Bilinmeyen sınıf, atlandı: {image_path}")
            continue

        # Telefon fotoğraflarındaki yan dönme (EXIF) düzeltilir
        original_image = ImageOps.exif_transpose(
            Image.open(image_path)
        ).convert("RGB")

        image_tensor = transform(
            original_image
        ).unsqueeze(0).to(device)

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        predicted_class = class_names[prediction.item()]
        confidence = confidence.item() * 100

        is_correct = predicted_class == true_class

        if is_correct:
            correct += 1

        total += 1

        # Ekranda göstermek için küçültülmüş kopya (tahmin orijinal görüntüyle yapıldı)
        display_image = original_image.copy()
        display_image.thumbnail((THUMB_SIZE, THUMB_SIZE))

        results.append({
            "path": image_path,
            "image": display_image,
            "true": true_class,
            "predicted": predicted_class,
            "confidence": confidence,
            "correct": is_correct
        })


# -----------------------------
# SONUÇLARI GÖRSELLEŞTİR
# -----------------------------

if total == 0:
    raise ValueError("Test edilecek geçerli fotoğraf bulunamadı!")

accuracy = correct / total * 100

os.makedirs("results", exist_ok=True)

# Sonuçlar PAGE_SIZE'lık sayfalara bölünür; her sayfa ekrana sığar
pages = [results[i:i + PAGE_SIZE] for i in range(0, total, PAGE_SIZE)]
saved_files = []

for page_number, page_results in enumerate(pages, start=1):

    n = len(page_results)
    columns = min(n, MAX_COLUMNS)
    rows = (n + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(CELL_WIDTH * columns, CELL_HEIGHT * rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, result in zip(axes, page_results):

        ax.imshow(result["image"])
        ax.axis("off")

        status = "DOĞRU" if result["correct"] else "YANLIŞ"

        title = (
            f"{result['path'].name}\n"
            f"Gerçek: {result['true']}\n"
            f"Tahmin: {result['predicted']}\n"
            f"Güven: %{result['confidence']:.1f}\n"
            f"{status}"
        )

        ax.set_title(
            title,
            fontsize=9,
            pad=6,
            color="green" if result["correct"] else "red"
        )

    # Kullanılmayan grafik alanlarını kapat
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Kendi Fotoğraflarımın Testi - Başarı: %{accuracy:.0f} ({correct}/{total})"
        f"   |   Sayfa {page_number}/{len(pages)}",
        fontsize=13,
        fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    file_name = f"results/my_photos_predictions_{page_number}.png"
    fig.savefig(file_name, dpi=150, bbox_inches="tight")
    saved_files.append(file_name)

print("\nKENDİ FOTOĞRAFLARIMIZIN TESTİ")
print("=" * 50)

for result in results:
    print(f"\nFotoğraf: {result['path'].name}")
    print(f"Gerçek sınıf: {result['true']}")
    print(f"Tahmin: {result['predicted']}")
    print(f"Güven: %{result['confidence']:.2f}")
    print("Sonuç:", "DOĞRU" if result["correct"] else "YANLIŞ")

print("\n" + "=" * 50)
print(f"Toplam fotoğraf: {total}")
print(f"Doğru tahmin: {correct}")
print(f"Yanlış tahmin: {total - correct}")
print(f"Başarı oranı: %{accuracy:.2f}")

print("\nGörseller kaydedildi:")
for file_name in saved_files:
    print(" -", file_name)

if SHOW_WINDOW:
    plt.show()