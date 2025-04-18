# 수정해야 할 점

### 1. 모델 저장 경로
- 현재는 모델 저장 경로가 코드에 고정값으로 주어져 있다.
- 추후 유연한 관리를 위해 config.yaml 혹은 .env로 관리

### 2. 모델 버전 관리
- 동일 경로에 모델 저장 시 과거 모델 덮어쓰기됨. (현재는 일괄 재학습하기 때문)
- 날짜 or 버전태그 포함한 경로 사용 추천 (용량 계속 커지기 때문에 관리 필요)

### 3. 모델 평가 결과 저장 방식
- 현재는 모델 별로 평가지표 .pkl에 저장, 별도의 DB에 연결하여 관리하면 추후 대시보드 시각화 등 비교분석에 이용하기 더 편할 것

### 4. 하이퍼파라미터 튜닝
- Optuna 이용하여 각 모델 별로 최적의 하이퍼파라미터 조합 도출 필요 (구현 경험 있음)

### 5. 학습 속도 최적화
- multiprocessing / joblib.Parallel / Ray 활용해서 병렬화 가능 (아직 잘 모르겠다. 조사 필요)

### 6. 데이터베이스 관리
SQLAlchemy는 ORM(Object Relational Mapping) 라이브러리 (JPA Hibernate처럼 관계형 DB를 객체로 관리하는 것 적용 필요)


### 참고사항 (조사 필요)
- Airflow:	데이터 파이프라인의 실행 순서(DAG)와 스케줄링, 상태 추적 / 	ETL 처리, 모델 학습, 예측, 실패 알림 등
- MLflow:	모델 실험 및 버전 관리 / 실험 결과 비교, 웹 UI로 추적 가능, 모델 저장소 역할
- Crontab: 	특정 시간마다 Python 스크립트를 자동 실행함 (예: 매일 새벽 3시) / 	주기적 재학습
