# Optimal-Stock-Forecasting
MFC 내 수요의 변동성을 고려한 주요 품목군 별 LSTM 기반 적정 재고 예측

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
        &emsp;"predicted_value": predicted_value, <br>
        &emsp;"safety_stock": safety_stock, <br>
        &emsp;"proper_stock": proper_stock, <br>
        &emsp;"precaution_comment": precaution_comment <br>
} <br>

# Easy usage with Docker Image
**docker run -p 8001:8001 kosonkh7/team4_stock_prediction:v0.0.0** (or latest) <br>
test url: localhost:8001/docs <br>

# Description

![image](https://github.com/user-attachments/assets/66e3ded8-553f-43cb-96e3-862c1958c4c2)


## Data Description
서울 열린 데이터 광장 [CJ 대한통운 택배 물동량 데이터.](https://data.seoul.go.kr/dataVisual/seoul/SeoulConsumerLogistics.do) <br>

18~23년 1달 단위로 집계된 데이터. 생활물류 11개 품목군에 대한 일자 별, 지역 별 물동량 정보. <br>

본 데이터를 목적에 맞게 집계 및 전처리하여 아래 csv 파일 형태로 변환하여 활용. <br>

- logistics_18_23.csv : [배송일자, 자치구] 기준으로 그룹핑하여, 자치구-자치구 + 전국시도-자치구 생활물류 데이터를 총합한 데이터, (결측치 처리 完)
- logistics_by_center.csv : 일자 별, 물류센터 별, 주요 품목군 별로 집계한 데이터.

## 3. 택배 물동량에 영향 미칠 요인
#### 시계열 특성
- 이동평균 (Moving Average)
- 지수평활 (Exponential Smoothing)
- 특정 기간 사이 표준편차 / 변동계수
- lagging
- rolling
- 차분

#### 날씨
- 기온, 강수량, 습도, 바람

#### 계절성, 공휴일
- 평일/주말, 월별, 분기별
- 공휴일 여부 (크리스마스, 발렌타인, 설, 추석.. 등)
- 이벤트 (블랙 프라이데이 등 할인)

#### 경제 지표
- 소득 수준
- 실업률
- 소비자 신뢰지수 등

  

## 4. 안전재고 계산식

안전 재고 계산에 필요한 항목: 
- **수요의 변동성**
- **리드 타임**
- **서비스 수준**


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


## 5. 고민해야할 점
- 극단적인 이상치가 있다면, 처리를 하는 것이 모델의 강건함을 더해줄 것.
- 품목군 별 물동량에 직접적으로 영향을 주는 이벤트(ex. 블랙프라이데이)를 피처로 추가하면 보다 정확한 예측 기대
- LSTM, GRU, RNN 총 3가지 모델의 예측 성능을 비교 평가하여 LSTM을 이용하였지만, 더 많은 시계열 예측 모델과의 비교 평가를 진행하면 좋았을 것.
[(ex. GluonTS : 딥러닝 기반 확률적 시계열 모델 패키지)](https://ts.gluon.ai/stable/index.html)
