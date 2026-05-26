# SKN31-2nd-2Team
# 🎵 Spotify 이용자 이탈 예측 머신러닝 프로젝트

## 팀원 및 역할분담

---

## 프로젝트 개요
본 프로젝트는 Spotify 이용자의 행동 패턴, 구독 정보, 이용 만족도 지표를 분석하여 이탈 가능성이 높은 고객을 사전에 식별하고, 이를 방지하기 위한 데이터 기반의 머신러닝 모델을 개발하는 것을 목표로 합니다.
---

## 프로젝트 배경
- 스트리밍 시장의 경쟁 심화: 유튜브 뮤직, 애플 뮤직 등 경쟁 서비스의 확대로 인해 신규 고객 유치 비용(CAC)이 기존 고객 유지 비용(CRC)보다 5~25배 더 높게 발생하고 있습니다.

- 고객 경험의 정량화 필요성: 단순히 '이용 중'인 상태를 넘어, 곡 건너뛰기 비율(Skip Rate)이나 청취 시간 등 실제 이용 행태가 이탈에 미치는 영향을 수치적으로 파악할 필요가 있습니다.

- 데이터 기반 의사결정: 축적된 유저 로그 데이터를 활용해 직관이 아닌 객관적인 지표로 이탈 징후를 포착하고자 합니다.
---

## 프로젝트 필요성
- 수익성 극대화: 이탈 가능성이 높은 'Family' 플랜 등 고단가 유저를 선제적으로 관리함으로써 매출 손실을 최소화합니다.

- 개인화된 리텐션 전략: 분석 결과 확인된 '높은 Skip Rate' 등 서비스 불만족 요소를 파악하여 유저별 맞춤형 추천 시스템 개선 및 프로모션 기획의 근거를 제공합니다.

- 운영 효율성 증대: 모든 유저가 아닌 이탈 고위험군에 마케팅 자원을 집중 투입하여 마케팅 효율(ROI)을 개선합니다.
---

## 핵심 분석 및 기대 효과
- 주요 인사이트:

   -  Family 플랜 유저가 타 요금제 대비 가장 높은 이탈률(27.52%)을 보이고 있어 집중 관리가 필요함이 확인되었습니다.

   - Skip Rate는 이탈과 가장 밀접한 상관관계를 보이며, 사용자 만족도를 대변하는 핵심 지표입니다.

- 기대 효과:

   - 예측 모델을 통한 이탈 징후 조기 포착 및 대응 프로세스 구축.

   - 이용자 몰입도(Listening Time) 향상을 위한 맞춤형 콘텐츠 제안 기반 마련.
---

## WBS

---
## Tech Stack

---

## 저장소 구조

```text
├── README.md                 # 프로젝트 소개 및 데이터 명세서
├── app/
│   └── app.py                    # streamlit 화면 구현
├── data/
│   └── spotify_churn_dataset.csv # 원본 데이터셋 (Raw Data)
├── notebooks/
│   ├── 01.eda.ipynb              # 탐색적 데이터 분석 및 시각화 코드
│   └── 02.preprocessing.ipynb    # 데이터 전처리 및 피처 엔지니어링 코드
│   └── 03.modeling               # 모델링 코드
├── models/
│   └── churn_predict_model.pkl   # 학습이 완료된 머신러닝 모델 파일
├── src/ 
│   └── .gitignore

```
---

## 데이터 명세서

본 문서는 Spotify 이용자 이탈 예측 머신러닝 모델 개발 프로젝트에 사용되는 원시 데이터(Raw Data)의 컬럼 정보와 전처리 가이드를 담고 있습니다.

데이터 출처 : [Spotify Analysis Dataset 2025 (Kaggle)](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)

### 데이터 구조 개요
- **데이터셋 파일명:** `spotify_churn_dataset.csv`
- **목적:** 사용자 행동 패턴 및 결제 데이터를 기반으로 한 이탈 여부(`is_churned`) 이진 분류(Binary Classification)
- **대상 변수 (Target):** `is_churned` (0: 유지, 1: 이탈)

---

### 컬럼별 상세 명세

| 컬럼명 (Column Name) | 데이터 타입 (Type) | 변수 유형 (Class) | 설명 (Description) | 예시 값 (Example) | 전처리 시 고려사항 및 비고 (Pre-processing Notes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **user_id** | 수치형 (Integer) | 식별자 (ID) | 고유 사용자 식별 번호 | `1`, `7971` | 단순 식별용 변수이므로 **모델 학습 시 제외(Drop)** 필요. |
| **gender** | 범주형 (String) | 독립 변수 (Feature) | 사용자의 성별 | `Female`, `Male`, `Other` | 다중 범주형 변수. **원-핫 인코딩(One-Hot Encoding)** 적용 필요. |
| **age** | 수치형 (Integer) | 독립 변수 (Feature) | 사용자의 나이 (만 나이) | `18`, `54`, `58` | 데이터 분포 확인 후 **스케일링(Scaling)** 또는 연령대별 **범주화(Binning)** 검토. |
| **country** | 범주형 (String) | 독립 변수 (Feature) | 거주 국가 (ISO 국가 코드 2자리) | `US`, `CA`, `DE`, `IN` | 범주가 많을 경우 고차원 방지를 위해 원-핫 인코딩 외에 **빈도수 기반 인코딩(Frequency/Target Encoding)** 고려. |
| **subscription_type**| 범주형 (String) | 독립 변수 (Feature) | 현재 이용 중인 구독 요금제 유형 | `Free`, `Premium`, `Family`, `Student` | 이탈률과 밀접한 핵심 변수. 순서형(Ordinal) 성격 여부 판단 후 인코딩 방식 결정. |
| **listening_time** | 수치형 (Integer) | 독립 변수 (Feature) | 총 청취 시간 (분 단위) | `26`, `141`, `280` | 연속형 수치 데이터. 이상치(Outlier) 제거 및 **MinMax/Standard 스케일링** 필요. |
| **songs_played_per_day**| 수치형 (Integer) | 독립 변수 (Feature) | 하루 평균 음악 재생 곡 수 | `3`, `23`, `62` | `listening_time`과 강한 상관관계를 보일 수 있으므로 **다중공선성(Multicollinearity)** 확인 필요. |
| **skip_rate** | 수치형 (Float) | 독립 변수 (Feature) | 곡 건너뛰기 비율 (0.0 ~ 1.0) | `0.04`, `0.20`, `0.46` | 이미 0과 1 사이로 정규화된 형태. 사용자 만족도를 나타내는 프록시(Proxy) 지표로 활용 가능. |
| **device_type** | 범주형 (String) | 독립 변수 (Feature) | 주로 사용하는 디바이스 플랫폼 | `Desktop`, `Mobile`, `Web` | 범주형 변수. **원-핫 인코딩(One-Hot Encoding)** 적용. |
| **ads_listened_per_week**| 수치형 (Integer) | 독립 변수 (Feature) | 주당 광고 청취 횟수 | `0`, `13`, `44` | `subscription_type`이 'Free'인 유저에게서만 높게 나타나는 경향 확인 및 교차 효과 분석 필요. |
| **offline_listening** | 범주형/이진 (Binary) | 독립 변수 (Feature) | 오프라인 다운로드/청취 기능 사용 여부 | `0` (미사용), `1` (사용) | 이미 0과 1로 이진 인코딩(Binary Encoded) 완료되어 변환 없이 수치 데이터로 바로 사용 가능. |
| **is_churned** | 범주형/이진 (Binary) | **종속 변수 (Target)** | 서비스 탈퇴(이탈) 여부 | `0` (유지), `1` (이탈) | **모델의 예측 목표.** 전체 데이터에서 0과 1의 비율을 확인하여 **클래스 불균형(Class Imbalance)** 대응 필요. |

---