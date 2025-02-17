import numpy as np
import pandas as pd

def calculate_safety_stock(data:pd.DataFrame, date:str, category_name:str):
    current_date = pd.to_datetime(date)
    three_months_ago = current_date - pd.DateOffset(months=3)
    recent_data = data[(data['date'] >= three_months_ago) & (data['date'] < current_date)]
    std_demand = recent_data[category_name].std()
    
    service_level = 1.28 # "food", "life", "baby", "cosmetic" # 90% 서비스 수준
    if category_name in ["book", "fashion", "goods", "sports", "digital", "other"]: 
        service_level = 0.841 # 80% 서비스 수준
    elif category_name == "furniture":
        service_level = 0.674 # 75% 서비스 수준

    safety_stock = service_level * std_demand * np.sqrt(3)
    safety_stock = int(safety_stock)
    
    return safety_stock
