from fastapi import APIRouter
from utils import load_data, feature_engineering, scale_data, calculate_safety_stock, create_precaution_comment, model_manager


router = APIRouter()

@router.get("/stock_predictions")
def predict(date: str, location: str, category_name: str):
    # 날짜 변환
    date = '2023' + date[4:]
    # 데이터, 모델, 스케일러 불러오기
    data, holiday = load_data(location, category_name)
    model = model_manager.load_model(location, category_name)
    scaler_X, scaler_y = model_manager.load_scalers(location, category_name)
    # 파생변수 생성
    data = feature_engineering(data, category_name, holiday)
    # 예측 데이터 선정 (입력 날짜 -> 7일 뒤 예측)
    predict_data = data.loc[data.date == date]
    
    # 학습에 이용한 스케일러로 데이터 스케일링
    X_input = scale_data(predict_data, scaler_X)
    
    # 예측 및 역스케일링
    predicted_scaled = model.predict(X_input)
    predicted_value = scaler_y.inverse_transform(predicted_scaled)
    predicted_value = int(predicted_value[0][0])

    # 안전재고 계산
    safety_stock = calculate_safety_stock(data, date, category_name)
    # 적정재고 산정
    proper_stock = safety_stock + predicted_value
    # 품목군별 주의사항 문구 출력
    precaution_comment = create_precaution_comment(category_name)

    return {
        "predicted_value": predicted_value,
        "safety_stock": safety_stock,
        "proper_stock": proper_stock,
        "precaution_comment": precaution_comment
    }
