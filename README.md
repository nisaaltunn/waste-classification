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

## Proje Yapısı
```
├── data_loader.py       # veri seti indirme, train/val/test ayrımı, augmentation
├── model.py             # ResNet18 modeli
├── train.py             # eğitim, test ve metriklerin hesaplanması
├── app.py               # Gradio arayüzü
├── test_my_photos.py    # kendi fotoğraflarımla test
├── models/              # eğitilmiş model (best_model.pth)
├── my_photos/           # telefonla çekilen test fotoğrafları (sınıf klasörleri)
├── results/             # confusion matrix, grafikler, metrics.json, test görselleri
└── requirements.txt
```

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python train.py          # modeli eğitir (veri seti otomatik indirilir)
python app.py            # Gradio arayüzünü başlatır
python test_my_photos.py # my_photos klasöründeki fotoğrafları test eder
```

Eğitilmiş model `models/best_model.pth` içinde hazır olduğu için eğitim yapmadan doğrudan
`python app.py` ile uygulamayı çalıştırabilirsiniz. Arayüz `http://127.0.0.1:7860` adresinde açılır.
