"""
API FastAPI cho dự đoán Conservation Status của cá sấu sử dụng model J-48
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import json
import os

app = FastAPI(
    title="Cá sấu Bảo tồn - API Dự đoán",
    description="API dự đoán tình trạng bảo tồn của cá sấu sử dụng model J-48",
    version="1.0.0"
)

# CORS middleware để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model và các encoder
MODEL_PATH = 'model_j48.pkl'
ENCODERS_DIR = 'encoders'

try:
    model = joblib.load(MODEL_PATH)
    sex_encoder = joblib.load(f'{ENCODERS_DIR}/sex_encoder.pkl')
    country_encoder = joblib.load(f'{ENCODERS_DIR}/country_encoder.pkl')
    habitat_encoder = joblib.load(f'{ENCODERS_DIR}/habitat_encoder.pkl')
    continent_encoder = joblib.load(f'{ENCODERS_DIR}/continent_encoder.pkl')
    target_encoder = joblib.load(f'{ENCODERS_DIR}/target_encoder.pkl')
    scaler = joblib.load(f'{ENCODERS_DIR}/scaler.pkl')
    
    # Load metadata và mappings
    with open('model_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open('mappings.json', 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    print("✅ Đã load model và các encoder thành công")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    model = None

# Mapping tiếng Việt cho Conservation Status
STATUS_VI = {
    'Least Concern': 'Ít quan ngại',
    'Vulnerable': 'Dễ bị tổn thương',
    'Critically Endangered': 'Cực kỳ nguy cấp',
    'Data Deficient': 'Thiếu dữ liệu',
    'Near Threatened': 'Gần bị đe dọa',
    'Endangered': 'Nguy cấp'
}

# Pydantic models cho request/response
class PredictionRequest(BaseModel):
    observed_length: float = Field(..., description="Chiều dài (m)", ge=0.1, le=10.0)
    observed_weight: float = Field(..., description="Cân nặng (kg)", ge=0.1, le=2000.0)
    age_class: str = Field(..., description="Nhóm tuổi")
    sex: str = Field(..., description="Giới tính")
    habitat_type: str = Field(..., description="Loại môi trường sống")
    country_region: Optional[str] = Field(None, description="Quốc gia/Khu vực")
    continent: str = Field(..., description="Khu vực")

class PredictionResponse(BaseModel):
    predicted_status: str
    tinh_trang_tieng_viet: str
    confidence_percent: float
    input_summary: dict

def encode_features(data: PredictionRequest) -> np.ndarray:
    """Mã hóa và chuẩn hóa features từ request"""
    # Xử lý Age Class
    age_mapping = mappings['age_mapping']
    age_encoded = age_mapping.get(data.age_class, 2)
    
    # Mã hóa Sex
    try:
        if data.sex in sex_encoder.classes_:
            sex_encoded = sex_encoder.transform([data.sex])[0]
        else:
            sex_encoded = 0  # Mặc định là giá trị đầu tiên
    except Exception:
        sex_encoded = 0
    
    # Mã hóa Country/Region
    country_value = data.country_region if data.country_region else ""
    try:
        if country_value and country_value in country_encoder.classes_:
            country_encoded = country_encoder.transform([country_value])[0]
        else:
            # Nếu không có country, tìm country đầu tiên trong continent
            # Đơn giản hóa: sử dụng giá trị mặc định
            country_encoded = 0
    except Exception:
        country_encoded = 0
    
    # Mã hóa Habitat Type
    try:
        if data.habitat_type in habitat_encoder.classes_:
            habitat_encoded = habitat_encoder.transform([data.habitat_type])[0]
        else:
            # Nếu habitat không có trong encoder, tìm giá trị gần nhất hoặc mặc định
            habitat_encoded = 0
    except Exception:
        habitat_encoded = 0
    
    # Mã hóa Continent
    try:
        if data.continent in continent_encoder.classes_:
            continent_encoded = continent_encoder.transform([data.continent])[0]
        else:
            continent_encoded = 0
    except Exception:
        continent_encoded = 0
    
    # Chuẩn hóa dữ liệu số
    numeric_features = np.array([[data.observed_length, data.observed_weight]])
    numeric_scaled = scaler.transform(numeric_features)
    
    # Kết hợp tất cả features theo thứ tự: length, weight, age, sex, country, habitat, continent
    features = np.array([[
        numeric_scaled[0][0],  # Observed Length (m)
        numeric_scaled[0][1],  # Observed Weight (kg)
        age_encoded,           # Age Class
        sex_encoded,           # Sex
        country_encoded,       # Country/Region
        habitat_encoded,       # Habitat Type
        continent_encoded      # Continent
    ]])
    
    return features

@app.get("/", response_class=HTMLResponse)
async def root():
    """Trang chủ - trả về file HTML"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>API đang chạy</h1><p>Truy cập <a href='/docs'>/docs</a> để xem API documentation</p>"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Dự đoán tình trạng bảo tồn của cá sấu
    
    - **observed_length**: Chiều dài quan sát được (m)
    - **observed_weight**: Cân nặng quan sát được (kg)
    - **age_class**: Nhóm tuổi (Hatchling, Juvenile, Subadult, Adult)
    - **sex**: Giới tính (Male, Female)
    - **habitat_type**: Loại môi trường sống
    - **country_region**: Quốc gia/Khu vực (tùy chọn)
    - **continent**: Khu vực địa lý
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load. Vui lòng train model trước.")
    
    try:
        # Mã hóa và chuẩn hóa features
        features = encode_features(request)
        
        # Dự đoán
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Lấy nhãn dự đoán
        predicted_status = target_encoder.inverse_transform([prediction])[0]
        
        # Tính độ tin cậy (xác suất cao nhất)
        confidence = float(max(probabilities))
        confidence_percent = round(confidence * 100, 2)
        
        # Mapping tiếng Việt
        tinh_trang_tieng_viet = STATUS_VI.get(predicted_status, predicted_status)
        
        # Tóm tắt input
        input_summary = {
            "Chiều dài": f"{request.observed_length} m",
            "Cân nặng": f"{request.observed_weight} kg",
            "Nhóm tuổi": request.age_class,
            "Giới tính": request.sex,
            "Môi trường": request.habitat_type,
            "Quốc gia": request.country_region or "N/A",
            "Khu vực": request.continent
        }
        
        return PredictionResponse(
            predicted_status=predicted_status,
            tinh_trang_tieng_viet=tinh_trang_tieng_viet,
            confidence_percent=confidence_percent,
            input_summary=input_summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.get("/health")
async def health():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": metadata.get('model_type', 'N/A') if model else None,
        "accuracy": metadata.get('accuracy', 0) if model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

