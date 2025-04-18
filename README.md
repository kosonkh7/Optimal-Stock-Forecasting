![image](https://github.com/user-attachments/assets/305e4e39-2faf-4896-aea2-b78f4ebd98f2)

- MFC 내 수요의 변동성을 고려한 주요 품목군 별 LSTM 기반 적정 재고 관리 서비스

# REST API developed using FastAPI
**Endpoint:** <br>
`GET /stock_predictions`

**Request**: <br>
{ <br>
&emsp;"date": str (예측일자) <br>
&emsp;"loaction": str (대상 센터) <br>
&emsp;"category_name": str (대상 품목군) <br>
} <br>

**Response**: <br>
{ <br>
        &emsp;"predicted_value": 예측 수요량 (LSTM 기반 예측 결과) <br>
        &emsp;"safety_stock": 안전 재고량 (수요의 변동성과 품목군의 특성을 반영한 안전재고 계산) <br>
        &emsp;"proper_stock": 적정 재고량 (예측수요 + 안전재고) <br>
        &emsp;"precaution_comment": 주의 문구 (품목군 별 재고 관리 시 유의 사항 메세지) <br>
} <br>

# Easy usage with Docker Image
**docker run -p 8001:8001 kosonkh7/team4_stock_prediction:v0.0.0** (or latest) <br>
test url: localhost:8001/docs <br>

# Directory Structure
📦재고 예측\
 ┣ 📂data\
 ┃ ┣ 📂raw\
 ┃ ┃ ┣ 📜dong.csv\
 ┃ ┃ ┣ 📜gu.csv\
 ┃ ┃ ┗ 📜people.csv\
 ┃ ┣ 📜holiday.csv\
 ┃ ┗ 📜logistics_by_center.csv\
 ┣ 📂model\
 ┃ ┣ 📂가락시장    (예시, 총 17개의 MFC)\
 ┃ ┃ ┣ 📜food.keras    (예시, 총 11개의 품목군)\
 ┃ ┃ ┣ 📜food_metrics.pkl\
 ┃ ┃ ┣ 📜food_scaler_X.pkl\
 ┃ ┃ ┣ 📜food_scaler_y.pkl\
 ┃ ┃ ┣ 📜sports.keras\
 ┃ ┃ ┣ 📜sports_metrics.pkl\
 ┃ ┃ ┣ 📜sports_scaler_X.pkl\
 ┃ ┃ ┗ 📜sports_scaler_y.pkl\
 ┣ 📂notebooks\
 ┃ ┣ 📜1. Preprocessing (seoul_to_seoul).ipynb\
 ┃ ┣ 📜2. Preprocessing (all_to_seoul).ipynb\
 ┃ ┣ 📜3. Preprocessing (split_unit_area).ipynb\
 ┃ ┣ 📜4. Holiday.ipynb\
 ┃ ┗ 📜5. GlounTS Estimator.ipynb\
 ┣ 📂routers\
 ┃ ┗ 📜stock_prediction.py\
 ┣ 📂training\
 ┃ ┗ 📜auto_training_pipeline.py\
 ┣ 📂utils\
 ┃ ┣ 📜data_loader.py\
 ┃ ┣ 📜feature_engineering.py\
 ┃ ┣ 📜model_loader.py\
 ┃ ┣ 📜precaution_comment.py\
 ┃ ┣ 📜safety_stock.py\
 ┃ ┣ 📜scaler.py\
 ┃ ┗ 📜__init__.py\
 ┣ 📜.dockerignore\
 ┣ 📜config.py\
 ┣ 📜Dockerfile\
 ┣ 📜main.py\
 ┗ 📜requirements.txt\

# Description

![image](https://github.com/user-attachments/assets/66e3ded8-553f-43cb-96e3-862c1958c4c2)


# Data Description
서울 열린 데이터 광장 [CJ 대한통운 택배 물동량 데이터.](https://data.seoul.go.kr/dataVisual/seoul/SeoulConsumerLogistics.do) <br>

18~23년 1달 단위로 집계된 데이터. 생활물류 11개 품목군에 대한 일자 별, 지역 별 물동량 정보. <br>

본 데이터를 목적에 맞게 집계 및 전처리하여 아래 csv 파일 형태로 변환하여 활용. <br>

- logistics_18_23.csv : [배송일자, 자치구] 기준으로 그룹핑하여, 자치구-자치구 + 전국시도-자치구 생활물류 데이터를 총합한 데이터. \
  (결측치 처리 完)
- logistics_by_center.csv : 일자 별, 물류센터 별, 주요 품목군 별로 집계한 데이터.
- holiday.csv : 물동량에 영향 크게 미치는 명절(신정, 설날, 추석) 데이터, 그외 공휴일 데이터를 포함하여 전처리 한 데이터.\
  (holidays 라이브러리 활용)

# Feature Engineering
- 시계열 특성: 주변 시점의 특성을 반영하기 위함 (7일/28일 단위 구분 -> Moving_Avg, Coeff_Var, Diff, Lag, Seasonality, Residual) <br>
- 날짜 특성: 택배 물동량은 특히 주말, 공휴일 여부에 크게 영향 받음 (연, 월, 요일, 주말, 공휴일, 명절 데이터) <br>

# Modeling
![image](https://github.com/user-attachments/assets/17e18d96-df70-4095-b35b-befcab904ce3)

LSTM: 입력 데이터가 길어질 때 기울기 소실이 발생한 것을 보완한 모델로, 재고 관리와 같은 시계열 예측에 특화된 모델.

샘플 데이터를 대상으로, 3가지 AI 기반 예측 모델(RNN, LSTM, GRU) 간의 성능 평가지표를 비교 했을 때

LSTM이 3가지 평가지표(R2, RMSE, MAE)에서 모두 좋은 성능을 보였기 때문에 채택하였다.

![image](https://github.com/user-attachments/assets/77cfa2bf-05a6-4617-b117-cadf885d5e94)

17개의 물류센터, 11개의 품목군 별로 나누어 총 187개의 독립적인 예측 모델을 학습하는 파이프라인을 개발하였고,

ReduceLROnPlateau, EarlyStopping를 통해 각 모델 성능을 최적화하였다.


## 안전재고 계산식

안전 재고 계산에 필요한 항목: 
- **수요의 변동성 (예측 시점 기준)**
- **품목군 별 리드 타임**
- **품목군 별 서비스 수준**


<details>
        <summary> <b>안전재고 계산식 상세 설명</b> </summary>     

#### 1. **일자별 수요 데이터**
특정 물품의 일별 수요량. 수요 데이터에서 평균 수요량과 변동성을 추정할 수 있다.

- **평균 수요 (Average Demand, D)**: 일정 기간 동안의 수요 평균값. (ex. 최근 30일) 
  
  $\[
  D = \frac{\text{총 수요}}{\text{일수}}
  \]$

- **수요의 표준편차 (Demand Standard Deviation, \(\sigma\))**: 일별 수요의 변동성.

  $\[
  \sigma = \sqrt{\frac{\sum (D_i - \text{평균 수요})^2}{N}}
  \]$
  
  여기서 $\( D_i \)$는 각 일자별 수요, $\( N \)$은 일자의 수.

#### 2. **리드 타임 (Lead Time, LT)**
리드 타임은 주문이 들어가서 물품이 입고되는 데 걸리는 시간. 이는 수요 예측을 기반으로 안전 재고를 설정하는데 중요한 요소이다.

- **리드 타임 평균 (Lead Time Average)**: 리드 타임 동안의 평균 수요를 계산.
  
  $\[
  D_{\text{lead time}} = D \times LT
  \]$

- **리드 타임의 표준편차**: 리드 타임 동안의 수요 변동성을 계산하여, 이를 바탕으로 안전 재고를 조정한다.

#### 3. **서비스 수준 (Service Level, SL)**
서비스 수준은 고객의 수요를 충족시키기 위해 필요한 재고의 확률적 목표. 예를 들어, 95%의 서비스 수준은 고객의 95%가 필요로 하는 제품을 확보하는 것을 의미한다. 서비스 수준에 따라 안전 재고의 양이 달라진다.

- **서비스 수준의 Z-점수**: 서비스 수준에 맞는 Z-점수를 사용하여, 수요의 변동성을 반영한 안전 재고를 계산할 수 있다. 예를 들어, 95% 서비스 수준에 해당하는 Z-점수는 약 1.65이다.

#### 4. **안전 재고 계산**
안전 재고는 주로 다음의 공식을 통해 계산할 수 있다:

$\[
\text{Safety Stock} = Z \times \sigma_{\text{LT}} \times \sqrt{LT}
\]$

- **Z**: 서비스 수준에 해당하는 Z-점수
- **$\(\sigma_{\text{LT}}\)$**: 리드 타임 동안의 수요 표준편차
- **$\(\sqrt{LT}\)$**: 리드 타임 동안의 수요 변동성을 고려

#### 5. **최종 안전 재고 공식**

$\[
\text{Safety Stock} = Z \times \sigma \times \sqrt{LT}
\]$

여기서:

- $\( \sigma \)$: 수요의 표준편차 (일자별 수요의 변동성)
- $\( LT \)$: 리드 타임 (일수)
- $\( Z \)$: 목표 서비스 수준에 해당하는 Z-점수

#### 예시
1. **수요 데이터**: 최근 30일 동안의 수요 평균은 100개, 표준편차는 20개
2. **리드 타임**: 5일
3. **서비스 수준**: 95% (Z-점수 = 1.65)

이 경우, 안전 재고는 다음과 같이 계산된다:

$\[
\text{Safety Stock} = 1.65 \times 20 \times \sqrt{5} \approx 73.65
\]$

따라서, 약 74개의 안전 재고를 유지해야 한다.

- 짧은 리드타임 (1-3일): 식품, 도서/음반
- 중간 리드타임 (3-7일): 기타, 생활/건강, 출산/육아, 패션의류, 패션잡화, 화장품/미용
- 긴 리드타임 (7-14일): 가구/인테리어, 디지털/가전, 스포츠/레저

</details>


## 고민해야할 점
- 극단적인 이상치가 있다면, 처리를 하는 것이 모델의 강건함을 더해줄 것.
- 품목군 별 물동량에 직접적으로 영향을 주는 이벤트(ex. 블랙프라이데이)를 추가하거나, 각 품목군별 특성을 세분화하여 새로운 피처를 정의했다면 보다 현실을 반영한 예측 가능했을 것.
- LSTM, GRU, RNN 총 3가지 모델의 예측 성능을 비교 평가하여 LSTM을 이용하였지만, 더 많은 시계열 예측 모델과의 비교 평가를 진행하면 좋았을 것.
[(ex. GluonTS : 딥러닝 기반 확률적 시계열 모델 패키지)](https://ts.gluon.ai/stable/index.html)
- Optuna와 같은 하이퍼파라미터 최적화 툴을 적용하여 최적의 모델 구조를 각각 도출해내는 파이프라인을 개발하면 좋을 것.
