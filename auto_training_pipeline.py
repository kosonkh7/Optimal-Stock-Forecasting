import pandas as pd
import numpy as np
import os
import joblib
import pickle

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout #, Embedding
from keras.callbacks import EarlyStopping
import keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.models import load_model


def feature_engineering(data:pd.DataFrame, holiday:pd.DataFrame, category_name:str):
    """_summary_
    시계열 특성 -> 주변 시점의 특성을 반영하기 위함 (이동평균, 변동계수, 차분, 시계열 분해 후 계절성, 잔차)
    날짜 특성 -> 택배 물동량은 특히 주말, 공휴일 여부에 크게 영향 받음 (연, 월, 요일, 주말, 공휴일, 명절)

    Args:
        data (pd.DataFrame): 특정 물류센터의 특정 품목에 대한 일자 별 물동량 정보 담긴 데이터프레임
        holiday (pd.DataFrame): 일자 별 공휴일 여부, 명절 여부 정보 담긴 데이터프레임
        category_name (str): 품목명

    Returns:
        피처 엔지니어링 된 데이터프레임
    """
    # 이동평균
    data['moving_avg_7'] = data[category_name].rolling(window=7).mean()
    data['moving_avg_28'] = data[category_name].rolling(window=28).mean()
    
    # 특정 기간 사이 표준편차 / 변동계수
    rolling_std = data[category_name].rolling(window=7).std()
    data['coeff_var_7'] = rolling_std / data['moving_avg_7']
    rolling_std = data[category_name].rolling(window=28).std()
    data['coeff_var_28'] = rolling_std / data['moving_avg_28']
    
    # Lagging
    data['lag_1'] = data[category_name].shift(1)
    data['lag_7'] = data[category_name].shift(7)
    
    # 차분
    data['diff_1'] = data[category_name].diff(1)
    data['diff_7'] = data[category_name].diff(7)
    
    # 시계열 분해
    result = seasonal_decompose(data[category_name], model='additive', period=7)
    data['trend_component'] = result.trend # 이동평균과 상관계수 높아서, 잔차 계산에만 쓰이고 제거
    data['seasonal_component'] = result.seasonal
    
    # 잔차
    data['residual'] = data[category_name] - (result.trend + data['seasonal_component'])
    data.drop(columns='trend_component', inplace=True) 
    
    # 지수평활 -> 이동평균과 상관계수 높아서 사용하지 않기로 결정정
    # data['exp_smoothing'] = ExponentialSmoothing(data[category_name], seasonal=None).fit().fittedvalues
    
    # 결측값 처리
    data = data.dropna()
    
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday  # 월요일=0, 일요일=6
    data['is_weekend'] = (data['date'].dt.weekday >= 5).astype(int)  # 5, 6: 주말(True)
    data = pd.merge(data, holiday, on='date', how='left')
    
    return data


def split_data(data:pd.DataFrame, category_name:str):
    """_summary_
    학습에 필요한 형태로 데이터 분리 및 전처리

    Args:
        data (pd.DataFrame): 피처엔지니어링 완료 된 데이터프레임
        category_name (str): 품목명

    Returns:
        LSTM 입력 형식에 맞게 3차원으로 구조 변환 된 X_train, X_test, y_train, y_test
    """
    columns_to_scale = ['moving_avg_7', 'moving_avg_28', 'coeff_var_7', 'coeff_var_28', 
                        'lag_1', 'lag_7', 'diff_1', 'diff_7', 'seasonal_component', 'residual']
    onehot = ['month', 'weekday', 'is_weekend', 'holiday', 'anniversary']
    
    # 7일 뒤 예측
    X = data[columns_to_scale+onehot][:-7]
    y = data[category_name].shift(-7).dropna().values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X[columns_to_scale]) # 예측할 타겟이 없는 부분 제외
    X_scaled = pd.DataFrame(X_scaled, columns=columns_to_scale) 
    X_scaled = pd.concat([X_scaled, X[onehot]], axis=1) # 스케일링 필요 없는 것 다시 이어 붙임
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=423, shuffle=False)
    
    # LSTM 입력 형식 변경
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y
        

def create_lstm_model(X_train):
    """_summary_
    LSTM 모델 생성과 콜백 함수(조기종료, 기울기 감소) 정의

    Args:
        X_train

    Returns:
        model, callbacks
    """
    K.clear_session()
    # LSTM 모델 정의
    model = Sequential()
    # 임베딩 레이어 추가 (month와 weekday에 대해 임베딩 적용)
    # model.add(Embedding(input_dim=12, output_dim=6, input_length=1, name='month_embedding'))
    # model.add(Embedding(input_dim=7, output_dim=3, input_length=1, name='weekday_embedding'))
    model.add(LSTM(units=512, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(units=1))
    # 조기 종료 및 학습률 감소 콜백 정의
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    # 모델 컴파일 및 학습
    model.compile(optimizer='adam', loss='mean_squared_error')
    callbacks = [early_stopping, reduce_lr]      
    
    return model, callbacks


def predict_model(model, X_test, scaler_y):
    """
    예측 결과에 대한 4가지 평가지표 (mse, rmse, mae, r2)를 딕셔너리 형태로 반환
    """
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    
    # 평가 지표 계산
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    
    metrics = {'mse':mse, 'rmse':rmse, 'mae':mae, 'r2':r2}
    
    return metrics
        

def save_model(model, location, category_name, scaler_X, scaler_y, metrics):
    """
    저장 항목: 모델(.keras), 스케일러(.pkl), 평가지표 딕셔너리(.pkl)
    저장 경로: model/location/category_name + @
    """
    # 경로 생성 및 모델 저장
    os.makedirs('model/'+location, exist_ok=True)
    model_name = 'model/'+location+'/'+category_name+'.keras'
    model.save(model_name)
    # 스케일러 저장
    joblib.dump(scaler_X, 'model/'+location+'/'+category_name+'_scaler_X.pkl')
    joblib.dump(scaler_y, 'model/'+location+'/'+category_name+'_scaler_y.pkl')
    # 평가지표 metrics 저장
    with open('model/'+location+'/'+category_name+'_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)



if __name__ == "__main__":
    # 택배 물동량 데이터, 공휴일 데이터 로드
    logistics = pd.read_csv("logistics_by_center.csv", encoding="euc-kr")
    logistics['date'] = pd.to_datetime(logistics['date'], format='%Y%m%d')
    holiday = pd.read_csv("holiday.csv", encoding='euc-kr')
    holiday['date'] = pd.to_datetime(holiday['date'], format='%Y%m%d')

    # 대상 물류센터(17개), 품목명(11개)
    location_list = ["가락시장", "고속터미널", "군자", "길음", "김포공항", "독산", "목동", "불광", "석계", 
                     "신내", "신림", "양재", "온수", "용산", "창동", "천호", "화계"]
    category_list = ["food", "life", "baby", "book", "cosmetic", "digital", "fashion", "furniture", "goods", "other", "sports"]
    
    # 총 187개의 학습 모델 생성
    for location in location_list:
        for category_name in category_list:
            # 물류센터 별, 품목 별 데이터 순차적으로 불러오기.
            data = logistics.loc[logistics.center_name==location][['date', category_name]].reset_index().drop(columns='index')
            # 예측 성능 향상을 위한 파생 변수 생성
            data = feature_engineering(data, holiday, category_name)
            # 데이터 스케일링 및 학습 가능한 형태로 변환
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_data(data, category_name)
            # 모델 생성. 콜백 함수 정의의
            model, callbacks = create_lstm_model(X_train)
            # 학습
            model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), 
                    callbacks=callbacks, verbose=1)    
            # 예측 결과 평가지표
            metrics = predict_model(model, X_test, scaler_y)
            # 경로에 맞게 모델, 스케일러, 예측 결과 저장
            save_model(model, location, category_name, scaler_X, scaler_y, metrics)
            
            print(f"{location} 센터의 {category_name} 수요 예측 모델 학습 완료.")