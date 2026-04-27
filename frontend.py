import streamlit as st
import torch
import torch.nn as nn
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



st.set_page_config(page_title="Fashion MNIST Classifier", layout="centered")
st.title('👕 Fashion MNIST Image Classifier')
st.write('Загрузите изображение предмета одежды, и нейросеть определит его категорию.')

# 3. Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_model():
    model = CheckImage()

    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Файл 'model.pth' не найден. Убедитесь, что вы сохранили модель после обучения.")
        return None


model = load_model()


classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


file = st.file_uploader('Выберите изображение...', type=['png', 'jpg', 'jpeg'])

if file is not None:

    image = Image.open(file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button('Распознать объект'):
        try:

            img_tensor = transform(image).unsqueeze(0).to(device)


            with torch.no_grad():
                output = model(img_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction] * 100


            st.success(f'Результат: **{classes[prediction]}**')
            st.info(f'Уверенность модели: {confidence:.2f}%')

        except Exception as e:
            st.error(f'Произошла ошибка при обработке: {e}')