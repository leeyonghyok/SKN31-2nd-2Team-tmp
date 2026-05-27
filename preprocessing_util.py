# preprocessing_util.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocessor():
    # ⚠️ 수정: 'CustomerID'와 'Churn'은 전처리 대상에서 제외해야 합니다.
    # ⚠️ 수정: 'TotalCharges'는 숫자형이므로 numeric에만 존재해야 합니다.
    categorical_columns = ["Gender", "Contract", "PaymentMethod"] 
    numeric_columns = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())  
    ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor_transformer = ColumnTransformer([
        ("cat", category_pipeline, categorical_columns), 
        ("num", numeric_pipeline, numeric_columns)
    ])

    return preprocessor_transformer

def preprocess_data(df, test_size=0.2, random_state=42):
    # 1. 원본 데이터 복사 (원본 훼손 방지)
    df_clean = df.copy()
    
    # 🌟 [요청하신 부분] 타겟 라벨 'Churn'을 0과 1로 변환
    # 대소문자 고정(strip/lower) 후 No -> 0, Yes -> 1로 매핑합니다.
    df_clean['Churn'] = df_clean['Churn'].astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1})
    
    # 2. 피처(X)와 타겟(y) 분리
    # 'CustomerID'와 'Churn'을 드롭합니다.
    x = df_clean.drop(['CustomerID', 'Churn'], axis=1)
    y = df_clean['Churn']
    
    # 3. 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    processor = preprocessor()

    # 4. 훈련 및 테스트 데이터 변환
    x_train_trans = processor.fit_transform(x_train)
    x_test_trans = processor.transform(x_test)

    # 5. SMOTE 적용
    smote = SMOTE(random_state=random_state)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_trans, y_train)

    return x_train_balanced, x_test_trans, y_train_balanced, y_test, processor