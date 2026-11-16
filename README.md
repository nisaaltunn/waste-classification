# Waste Classification with ResNet18

Bu proje, atık görsellerini derin öğrenme kullanarak 6 farklı sınıfa ayırır.

## Sınıflar
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Veri Seti
TrashNet veri seti kullanılmıştır.
https://github.com/garythung/trashnet

Veri seti eğitim, doğrulama ve test kümelerine ayrılmıştır.

## Model
- Mimari: ResNet18
- Giriş boyutu: 224x224
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Epoch: 15
- Batch size: 32

## Test Sonuçları
- Test Accuracy: 92.89%
- Weighted Precision: 93.05%
- Weighted Recall: 92.89%
- Weighted F1-score: 92.91%

## Kendi Fotoğraflarımla Test
Model, telefonla çekilmiş 21 farklı fotoğraf üzerinde de test edilmiştir.
- Doğru tahmin: 17
- Yanlış tahmin: 4
- Başarı: 80.95%

## Kullanılan Teknolojiler
- Python
- PyTorch
- Torchvision
- Gradio
- Matplotlib
- Scikit-learn

## Çalıştırma
```bash
pip install -r requirements.txt
python app.py


