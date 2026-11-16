
import torch
import torch.nn as nn
from torchvision import models


class WasteCNN(nn.Module):

    def __init__(self, num_classes=6, pretrained=True):

        super(WasteCNN, self).__init__()

        # ImageNet üzerinde önceden eğitilmiş ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.model = models.resnet18(weights=weights)

        # Son sınıflandırma katmanını 6 sınıfa göre düzenle
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Modeli doğrudan çalıştırarak test et
if __name__ == "__main__":

    model = WasteCNN(num_classes=6, pretrained=False)

    sample = torch.randn(2, 3, 224, 224)

    output = model(sample)

    print("ResNet18 modeli başarıyla oluşturuldu!")
    print("Girdi boyutu:", sample.shape)
    print("Çıktı boyutu:", output.shape)

