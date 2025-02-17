import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_data(predict_data, scaler_X):
    # 학습 시와 동일하게 스케일링
    columns_to_scale = ['moving_avg_7', 'moving_avg_28', 'coeff_var_7', 'coeff_var_28', 
                        'lag_1', 'lag_7', 'diff_1', 'diff_7', 'seasonal_component', 'residual']
    onehot = ['month', 'weekday', 'is_weekend', 'holiday', 'anniversary']
    
    scaled_features = scaler_X.transform(predict_data[columns_to_scale])
    scaled_data = pd.DataFrame(scaled_features, columns=columns_to_scale)
    scaled_data = pd.concat([scaled_data, predict_data[onehot].reset_index(drop=True)], axis=1)
    X_input = np.reshape(scaled_data, (scaled_data.shape[0], scaled_data.shape[1], 1)) 
    
    return X_input