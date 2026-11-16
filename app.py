
import torch
import torch.nn.functional as F
import gradio as gr

from PIL import Image
from torchvision import transforms

from model import WasteCNN


# -----------------------------
# DEVICE
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Kullanılan cihaz:", device)


# -----------------------------
# EĞİTİLMİŞ MODELİ YÜKLE
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

print("Eğitilmiş model başarıyla yüklendi!")


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
# TAHMİN FONKSİYONU
# -----------------------------

def predict_waste(image):

    if image is None:
        return {}, "Lütfen bir fotoğraf yükleyin."

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)[0]

    results = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    predicted_index = torch.argmax(probabilities).item()

    predicted_class = class_names[predicted_index]

    confidence = probabilities[predicted_index].item() * 100

    message = (
        f"Tahmin edilen atık türü: {predicted_class}\n"
        f"Güven oranı: %{confidence:.2f}"
    )

    return results, message


# -----------------------------
# GRADIO ARAYÜZÜ
# -----------------------------

with gr.Blocks(title="Waste Classification") as app:

    gr.Markdown(
        """
        # ♻️ Atık Türü Sınıflandırma Uygulaması

        **ResNet18 Derin Öğrenme Modeli**

        Bir atık fotoğrafı yükleyin. Model fotoğrafı
        altı farklı atık sınıfından birine sınıflandıracaktır.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Atık Fotoğrafı"
            )

            predict_button = gr.Button(
                "Fotoğrafı Sınıflandır",
                variant="primary"
            )

        with gr.Column():

            prediction_output = gr.Label(
                num_top_classes=6,
                label="Sınıflandırma Sonuçları"
            )

            text_output = gr.Textbox(
                label="Tahmin Detayı",
                lines=3
            )

    predict_button.click(
        fn=predict_waste,
        inputs=image_input,
        outputs=[prediction_output, text_output]
    )

    gr.Markdown(
        """
        ---
        **Model Bilgileri**

        - Model: ResNet18
        - Görüntü boyutu: 224 × 224
        - Sınıf sayısı: 6
        - Veri seti: TrashNet

        **Sınıflar:** cardboard, glass, metal, paper, plastic, trash
        """
    )


# -----------------------------
# UYGULAMAYI BAŞLAT
# -----------------------------

if __name__ == "__main__":

    app.launch()

