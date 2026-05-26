# preprocessing_util.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocessor():
    categorical_columns = ['gender', 'country', 'subscription_type', 'device_type'] 
    numeric_columns = ['age', 'listening_time', 'songs_played_per_day', 'skip_rate', 'ads_listened_per_week', 'offline_listening', 'avg_time_per_song', 'underutilized_premium']

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
    x = df.drop(['user_id', 'is_churned'], axis=1)
    y = df['is_churned']
    
    # 1. 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    processor = preprocessor()

    # 2. 훈련 및 테스트 데이터 변환 (스케일링 & 인코딩 완료)
    x_train_trans = processor.fit_transform(x_train)
    x_test_trans = processor.transform(x_test)

    # 🌟 [SMOTE 적용 핵심 부위] 🌟
    # 변환이 완료되어 숫자로만 이루어진 x_train_trans 데이터에 SMOTE를 적용합니다.
    smote = SMOTE(random_state=random_state)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_trans, y_train)

    # 기존 x_train_trans, y_train 대신 밸런스가 맞춤형 변환된 데이터를 반환합니다.
    return x_train_balanced, x_test_trans, y_train_balanced, y_test, processor