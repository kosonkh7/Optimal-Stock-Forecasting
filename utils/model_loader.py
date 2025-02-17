import joblib
from tensorflow.keras.models import load_model
from config import MODEL_DIR

class ModelManager:
    def __init__(self):
        self.models = {}  # 모델 캐싱
        self.scalers = {}  # 스케일러 캐싱

    def load_model(self, location, category_name):
        key = f"{location}_{category_name}"
        if key not in self.models:
            print(f"🔄 Loading model for {location} - {category_name}...")
            self.models[key] = load_model(f"{MODEL_DIR}/{location}/{category_name}.keras")
        return self.models[key]

    def load_scalers(self, location, category_name):
        key = f"{location}_{category_name}"
        if key not in self.scalers:
            print(f"🔄 Loading scalers for {location} - {category_name}...")
            scaler_X = joblib.load(f"{MODEL_DIR}/{location}/{category_name}_scaler_X.pkl")
            scaler_y = joblib.load(f"{MODEL_DIR}/{location}/{category_name}_scaler_y.pkl")
            self.scalers[key] = (scaler_X, scaler_y)
        return self.scalers[key]

model_manager = ModelManager()