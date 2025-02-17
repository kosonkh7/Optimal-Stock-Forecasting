import os
from dotenv import load_dotenv

# .env 파일 로드 (Azure 환경에서 활용 가능)
if os.path.exists(".env"):
    load_dotenv()

# 데이터 및 모델 디렉토리 경로 설정
DATA_DIR = os.getenv("DATA_DIR", "data/")
MODEL_DIR = os.getenv("MODEL_DIR", "model/")