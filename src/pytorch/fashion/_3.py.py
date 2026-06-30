import io
import os
import sqlite3
from datetime import datetime
import uuid
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
CLASS_NAMES = ["bags", "jeans", "shoes", "t_shirt"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
WEIGHTS_PATH = "best_product_model.pth"
DB_PATH = "predictions_history.db"
TRAIN_DATASET_DIR = "my_dataset/train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predict_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(title="Product Classifier API with Data Collection")
model = None

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ И МОДЕЛИ
# ==========================================
def init_db():
    """Создает таблицу истории, если она еще не создана."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            original_filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            saved_path TEXT
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    global model
    init_db()
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Критическая ошибка: Файл весов '{WEIGHTS_PATH}' не найден!")
        return

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"--> Сервер запущен. Модель на устройстве: {device}")

# ==========================================
# 3. ОСНОВНОЙ МАРШРУТ (ПРЕДСКАЗАНИЕ + СБОР ДАННЫХ)
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        # 1. Чтение и подготовка картинки для нейросети
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_img = predict_transforms(image).unsqueeze(0).to(device)
        
        # 2. Предсказание нейросети
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            predicted_class = CLASS_NAMES[predicted_idx]

        # 3. АВТОМАТИЧЕСКИЙ СБОР ДАННЫХ: Сохранение файла в папку train
        # Генерируем уникальное имя файла, чтобы избежать перезаписи совпадений
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        new_filename = f"{unique_id}{file_extension}"
        
        # Целевая папка для этого класса (например, my_dataset/train/shoes/)
        target_dir = os.path.join(TRAIN_DATASET_DIR, predicted_class)
        os.makedirs(target_dir, exist_ok=True)
        
        saved_path = os.path.join(target_dir, new_filename)
        image.save(saved_path)

        # 4. СОХРАНЕНИЕ ИСТОРИИ В БАЗУ ДАННЫХ SQLITE
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.conn = conn.cursor()
        cursor.execute(
            "INSERT INTO history (id, timestamp, original_filename, predicted_class, confidence, saved_path) VALUES (?, ?, ?, ?, ?, ?)",
            (unique_id, timestamp, file.filename, predicted_class, round(confidence, 4), saved_path)
        )
        conn.commit()
        conn.close()

        # 5. Ответ клиенту
        return JSONResponse(content={
            "success": True,
            "id": unique_id,
            "result": {
                "class": predicted_class,
                "confidence_percent": round(confidence * 100, 2)
            },
            "data_collection": {
                "saved_for_training": True,
                "destination_folder": f"train/{predicted_class}"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# ==========================================
# 4. ДОПОЛНИТЕЛЬНЫЙ МАРШРУТ: ПРОСМОТР ЛОГОВ
# ==========================================
@app.get("/history")
def get_history(limit: int = 50):
    """Возвращает список последних сохраненных предсказаний из БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Чтобы получить данные в виде словаря
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
