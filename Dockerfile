# Python Slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 패키지 설치 (텐서플로우 및 기타 패키지 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    gcc \
    gfortran \
    wget \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libjpeg-dev \
    zlib1g-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* # 캐시 제거하여 이미지 크기 최적화

# pip, setuptools 최신화
RUN python -m pip install --upgrade pip setuptools wheel

# requirements.txt 복사
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출
EXPOSE 8001

ENV DATA_DIR=/app/data/
ENV MODEL_DIR=/app/model/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--log-level", "info"]