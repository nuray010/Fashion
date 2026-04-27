import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import io


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('fashion_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))


    img_tensor = transform(img).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()

    return {
        "class_id": prediction,
        "class_name": CLASSES[prediction]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)