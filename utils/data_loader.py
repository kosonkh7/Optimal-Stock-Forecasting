import pandas as pd
from config import DATA_DIR

def load_data(location: str, category_name: str):
    logistics = pd.read_csv(f"{DATA_DIR}/logistics_by_center.csv", encoding="euc-kr", usecols=['date', 'center_name', category_name])
    logistics['date'] = pd.to_datetime(logistics['date'], format='%Y%m%d')
    holiday = pd.read_csv(f"{DATA_DIR}/holiday.csv", encoding='euc-kr')
    holiday['date'] = pd.to_datetime(holiday['date'], format='%Y%m%d')
    data = logistics.loc[logistics.center_name == location][['date', category_name]].reset_index(drop=True)
    
    return data, holiday