import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def feature_engineering(data: pd.DataFrame, category_name: str, holiday: pd.DataFrame):
    # 이동평균
    data['moving_avg_7'] = data[category_name].rolling(window=7).mean()
    data['moving_avg_28'] = data[category_name].rolling(window=28).mean()
    
    # 최근 1주/1달 간 표준편차 / 변동계수
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
    data['seasonal_component'] = result.seasonal
    # 잔차
    data['residual'] = data[category_name] - (result.trend + data['seasonal_component'])
    
    data.dropna(inplace=True)
    
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday  # 월요일=0, 일요일=6
    data['is_weekend'] = (data['date'].dt.weekday >= 5).astype(int)  # 5, 6: 주말(True)
    data = pd.merge(data, holiday, on='date', how='left')

    return data
