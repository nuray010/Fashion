# 👗 Fashion MNIST Classifier

> Нейросеть на **PyTorch** для классификации изображений одежды по 10 категориям.  
> REST API на **FastAPI** + интерактивный веб-интерфейс на **Streamlit**.

---

## 🧠 Как это работает

Проект использует свёрточную нейронную сеть (CNN), обученную на датасете **Fashion MNIST**.  
Загружаешь изображение — модель определяет, что на нём: футболка, платье, кроссовки и т.д.

```
Изображение → Preprocessing (grayscale, 28×28) → CNN → Предсказание класса
```

---

## 🗂️ Классы одежды

| ID | Класс |
|---|---|
| 0 | 👕 T-shirt / Top |
| 1 | 👖 Trouser |
| 2 | 🧥 Pullover |
| 3 | 👗 Dress |
| 4 | 🧥 Coat |
| 5 | 👡 Sandal |
| 6 | 👔 Shirt |
| 7 | 👟 Sneaker |
| 8 | 👜 Bag |
| 9 | 👢 Ankle Boot |

---

## 🏗️ Архитектура модели

```
CheckImage (CNN)
├── Block 1
│   ├── Conv2d(1 → 16, kernel=3, padding=1)
│   ├── ReLU
│   └── MaxPool2d(2)
└── Block 2
    ├── Flatten
    ├── Linear(16×14×14 → 64)
    ├── ReLU
    └── Linear(64 → 10)
```

Входное изображение конвертируется в **grayscale** и масштабируется до **28×28 пикселей**.

---

## 📁 Структура проекта

```
Fashion/
├── main.py             # FastAPI — REST API для предсказаний
├── frontend.py         # Streamlit — веб-интерфейс
└── fashion_model.pth   # Обученные веса модели
```

---

## 🚀 Запуск

### Установка зависимостей

```bash
pip install fastapi uvicorn torch torchvision pillow streamlit
```

### 1. Запуск API (FastAPI)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Документация: `http://localhost:8000/docs`

### 2. Запуск веб-интерфейса (Streamlit)

```bash
streamlit run frontend.py
```

Откроется в браузере по адресу: `http://localhost:8501`

---

## 🔌 API Reference

### `POST /predict`

Загрузи изображение — получи предсказание.

**Request:**
```
Content-Type: multipart/form-data
file: <image file>
```

**Response:**
```json
{
  "class_id": 7,
  "class_name": "Sneaker"
}
```

**Пример через curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@my_shoe.jpg"
```

---

## 🖥️ Веб-интерфейс (Streamlit)

Streamlit-приложение позволяет:

- загрузить изображение (`jpg`, `png`, `jpeg`)
- нажать кнопку **«Распознать объект»**
- увидеть название класса и **уверенность модели** в процентах

---

## ⚙️ Технологии

| | Стек |
|---|---|
| **Deep Learning** | PyTorch |
| **Computer Vision** | torchvision, Pillow |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Датасет** | Fashion MNIST |

---

## 📌 Требования

- Python 3.9+
- PyTorch (CPU или CUDA)
- Файл весов `fashion_model.pth` в корне проекта

> Модель автоматически использует **GPU** (CUDA), если он доступен, иначе работает на **CPU**.

---

## 👤 Автор

**nuray010** — [github.com/nuray010](https://github.com/nuray010)
