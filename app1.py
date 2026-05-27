from __future__ import annotations

import os
import html
import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing_util
import imblearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COLUMN_KOR = {
    'CustomerID': '사용자 ID',
    'Age': '나이',
    'Gender': '성별',
    'Tenure': '가입 기간',
    'MonthlyCharges': '월 요금',
    'Contract': '계약 유형',
    'PaymentMethod': '결제 방법',
    'TotalCharges': '총 요금',
    'Churn': '이탈 여부',
}

def kor_col(col: str) -> str:
    if col in COLUMN_KOR:
        return COLUMN_KOR[col]
    if "_" in col:
        base, value = col.rsplit("_", 1)
        if base in COLUMN_KOR:
            return f"{COLUMN_KOR[base]}={value}"
    return col

def display_col(col: str) -> str:
    korean = kor_col(col)
    return f"{col} ({korean})" if korean != col else col

def step_title(title: str, done: bool = False) -> None:
    badge = "<span class='done-badge'>✓ 완료</span>" if done else ""
    st.markdown(f"<div class='step-title'><h4>{title}</h4>{badge}</div>", unsafe_allow_html=True)

def substep_title(title: str) -> None:
    st.markdown(f"<div class='substep-title'>{title}</div>", unsafe_allow_html=True)

def reset_model_flow_state() -> None:
    keys = [
        "best_params", "best_score", "best_model_name", "cv_results", "pending_manual_params",
        "pipe", "trained_model_name", "params", "X_test", "y_test", "pred", "proba",
        "train_metrics", "train_settings",
    ]
    for key in keys:
        st.session_state.pop(key, None)

def reset_data_flow_state() -> None:
    reset_model_flow_state()
    for key in ["model_name", "model_flow_model_name", "eda_cat", "eda_num"]:
        st.session_state.pop(key, None)


st.set_page_config(
    page_title="Spotify Churn Prediction",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp, .main { background: #f5f7fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, p, label, span, div { font-family: 'Segoe UI', sans-serif; }
    h1 { color: #102a1d; font-weight: 900; letter-spacing: -1px; }
    h2, h3, h4, p, label, span, div { color: #172033; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: #ffffff; padding: 8px; border-radius: 18px; border: 1px solid #d8e0ea; }
    .stTabs [data-baseweb="tab"] { height: 44px; border-radius: 14px; color: #334155; font-weight: 700; padding: 0 16px; }
    .stTabs [aria-selected="true"] { background: #1DB954; color: #06130b !important; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d8e0ea; padding: 16px; border-radius: 18px; box-shadow: 0 10px 26px rgba(15,23,42,.10); }
    div[data-testid="stMetric"] label { color: #475569 !important; }
    div[data-testid="stMetricValue"] { color: #0f172a !important; }
    .glass { background: #ffffff; border: 1px solid #d8e0ea; border-radius: 18px; padding: 20px; box-shadow: 0 10px 26px rgba(15,23,42,.10); color:#172033; }
    .hero { background: linear-gradient(135deg, #11813c 0%, #1DB954 45%, #0f5f32 100%); border:1px solid #0f6b37; border-radius:46px; padding:30px 32px; margin: 18px 0 22px 0; box-shadow: 0 14px 34px rgba(15,95,50,.22); overflow: hidden; box-sizing: border-box; }
    .hero-title { font-size: 2rem; line-height:1.15; color:#ffffff; font-weight:900; text-shadow: 0 2px 12px rgba(0,0,0,.20); word-break: keep-all; }
    .hero-sub { color:#e8fff0; font-size:1rem; margin-top:10px; font-weight: 700; }
    .small-note { color:#d7ffe3; font-size:.9rem; font-weight: 700; }
    .step-title { display:flex; align-items:center; gap:10px; margin: 1.1rem 0 .5rem 0; }
    .step-title h4 { margin:0; font-size:1.35rem; font-weight:900; color:#0f172a; }
    .done-badge { display:inline-flex; align-items:center; gap:6px; background:#dcfce7; color:#166534; border:1px solid #86efac; border-radius:999px; padding:4px 10px; font-size:.85rem; font-weight:800; }
    .substep-title { margin: 1rem 0 .45rem 0; font-size:1.05rem; font-weight:800; color:#1f2937; }
    .risk-high { background: #fee2e2; border:1px solid #ef4444; color:#7f1d1d; padding:16px; border-radius:18px; }
    .risk-low { background: #dcfce7; border:1px solid #22c55e; color:#14532d; padding:16px; border-radius:18px; }
    .stDataFrame { border-radius: 18px; overflow:hidden; }
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "data/synthetic_customer_churn_100k.csv"
RANDOM_STATE = 42

@st.cache_data
def load_data(path: str, file_mtime: float) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"{path} 파일이 없습니다. streamlit_app.py와 같은 폴더에 CSV를 넣어주세요.")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def is_integer_series(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    return pd.api.types.is_integer_dtype(clean) or bool(np.all(np.equal(np.mod(clean, 1), 0)))

def detect_target(df: pd.DataFrame) -> str:
    if "Churn" in df.columns:
        return "Churn"
    st.error("정답(label) 컬럼을 찾지 못했습니다. CSV에는 반드시 Churn 컬럼이 있어야 합니다.")
    st.write("현재 컬럼", df.columns.tolist())
    st.stop()

def split_columns(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str], list[str]]:
    drop_cols = [target_col]
    id_cols = [c for c in df.columns if c.lower().endswith("ID") or c.lower() in {"CustomerID", "id"}]
    feature_cols = [c for c in df.columns if c not in drop_cols + id_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", make_ohe(), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def get_feature_names(pipe: Pipeline, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    names = []
    pre = pipe.named_steps["preprocess"]
    
    # preprocessing 구조에 맞게 순서대로 추출
    if categorical_cols:
        # cat 트리 내부의 파이프라인에서 'ohe' 단계를 추출합니다.
        cat_pipeline = pre.named_transformers_["cat"]
        ohe = cat_pipeline.named_steps["ohe"]
        names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
        
    if numeric_cols:
        names.extend(numeric_cols)
    return names

def build_model(model_name: str, params: dict[str, Any], *, svm_probability: bool = True):
    if model_name == "Logistic Regression":
        return LogisticRegression(C=params["C"], max_iter=1000, random_state=RANDOM_STATE)
    if model_name == "SGD Classifier":
        return SGDClassifier(alpha=params["alpha"], loss="log_loss", max_iter=1000, random_state=RANDOM_STATE)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    if model_name == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"], probability=svm_probability, random_state=RANDOM_STATE)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], random_state=RANDOM_STATE)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=params["max_depth"], random_state=RANDOM_STATE)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], random_state=RANDOM_STATE)
    raise ValueError(model_name)

def params_ui(model_name: str, prefix: str = "manual") -> dict[str, Any]:
    params: dict[str, Any] = {}
    if model_name in ["Logistic Regression", "SVM"]:
        params["C"] = st.slider("C - 규제 강도 역수", 0.01, 10.0, 1.0, 0.01, key=f"{prefix}_C")
    if model_name == "SVM":
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"], key=f"{prefix}_kernel")
    if model_name == "SGD Classifier":
        params["alpha"] = st.slider("alpha - 규제 강도", 0.0001, 0.1, 0.001, 0.0001, format="%.4f", key=f"{prefix}_alpha")
    if model_name == "KNN":
        params["n_neighbors"] = st.slider("n_neighbors", 1, 25, 5, key=f"{prefix}_knn")
    if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        params["max_depth"] = st.slider("max_depth", 1, 30, 5, key=f"{prefix}_depth")
    if model_name in ["Decision Tree", "Random Forest"]:
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, key=f"{prefix}_split")
    if model_name in ["Random Forest", "Gradient Boosting", "AdaBoost"]:
        params["n_estimators"] = st.slider("n_estimators", 20, 300, 100, 10, key=f"{prefix}_n")
    if model_name in ["Gradient Boosting", "AdaBoost"]:
        params["learning_rate"] = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01, key=f"{prefix}_lr")
    return params

def apply_best_params_to_manual_widgets() -> None:
    pending = st.session_state.pop("pending_manual_params", None)
    if not pending:
        return
    param_key_map = {
        "model__C": "manual_C",
        "model__kernel": "manual_kernel",
        "model__n_neighbors": "manual_knn",
        "model__max_depth": "manual_depth",
        "model__min_samples_split": "manual_split",
        "model__n_estimators": "manual_n",
        "model__learning_rate": "manual_lr",
    }
    for grid_key, widget_key in param_key_map.items():
        if grid_key in pending:
            st.session_state[widget_key] = pending[grid_key]

def format_best_params_text(model_name: str, best_params: dict[str, Any], best_score: float) -> str:
    lines = [
        f"모델: {model_name}",
        "최적 하이퍼파라미터:",
    ]
    for key, value in best_params.items():
        lines.append(f"- {key.replace('model__', '')}: {value}")
    lines.append(f"교차검증 F1 점수: {best_score:.4f}")
    return "\n".join(lines)

def show_best_params_box(model_name: str, best_params: dict[str, Any], best_score: float) -> None:
    text = html.escape(format_best_params_text(model_name, best_params, best_score))
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #cbd5e1;
            border-left:6px solid #1DB954;
            border-radius:16px;
            padding:18px 20px;
            color:#0f172a;
            font-size:1rem;
            line-height:1.65;
            box-shadow:0 10px 24px rgba(15,23,42,.08);
            white-space:pre-wrap;
            font-family:'Segoe UI', sans-serif;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )

def format_param_dict(params: dict[str, Any]) -> str:
    return ", ".join(f"{key.replace('model__', '')}={value}" for key, value in params.items())

def show_best_params_area() -> None:
    st.markdown("#### 최적 하이퍼파라미터 결과")
    if "best_params" not in st.session_state:
        st.info("GridSearchCV를 실행하면 가장 성능이 좋았던 모델 설정이 여기에 텍스트로 표시됩니다. 이 값은 참고용이며, 아래 설정 영역의 슬라이더를 자동으로 바꾸지 않습니다.")
        return
    show_best_params_box(
        st.session_state["best_model_name"],
        st.session_state["best_params"],
        float(st.session_state["best_score"]),
    )

def predict_probability(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = pipe.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return pipe.predict(X).astype(float)

def compact_fig(width=4.8, height=2.8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.tick_params(labelsize=8)
    return fig, ax

def style_axis(ax):
    ax.grid(alpha=0.18)
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# 데이터 준비
uploaded_csv = st.session_state.get("uploaded_csv")
if uploaded_csv:
    df = load_uploaded_csv(uploaded_csv["bytes"])
    data_source_name = uploaded_csv["name"]
    data_source_caption = "업로드한 CSV"
else:
    df = load_data(DATA_PATH, os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0)
    data_source_name = DATA_PATH
    data_source_caption = "기본 CSV"

target_col = detect_target(df)
feature_cols, numeric_cols, categorical_cols = split_columns(df, target_col)
if not feature_cols:
    st.error("학습에 사용할 특성이 없습니다. Churn을 제외한 특성 컬럼이 최소 1개 필요합니다.")
    st.stop()
X = df[feature_cols]
y = df[target_col]
if y.dtype == "object":
    y = y.astype("category").cat.codes

apply_best_params_to_manual_widgets()


# <div class="small-note">사용 데이터: <b>{DATA_PATH}</b>
# <div class="small-note">데이터 출처: <b>Kaggle</b>
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">🎧 Spotify 고객 이탈 예측</div>
      
    </div>
    """,
    unsafe_allow_html=True,
)

model_trained = "pipe" in st.session_state
tab_labels = ["1. 파일 정보", "2. EDA", "3. 전처리", "4. 모델 학습"]
if model_trained:
    tab_labels.extend(["5. 평가 지표", "6. 모델 해석", "7. Spotify 고객 이탈 예측"])
else:
    tab_labels.extend(["5. 평가 지표 (비활성화)", "6. 모델 해석 (비활성화)", "7. Spotify 고객 이탈 예측 (비활성화)"])
tabs = st.tabs(tab_labels)

### 1. 데이터 정보 ###
with tabs[0]:
    st.markdown("### 파일 정보")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            preview_df = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
        except Exception as exc:
            st.error(f"CSV를 읽을 수 없습니다: {exc}")
        else:
            if "Churn" not in preview_df.columns:
                st.error("업로드한 CSV에는 정답(label) 컬럼인 Churn가 반드시 있어야 합니다.")
            else:
                current_upload = st.session_state.get("uploaded_csv")
                if (
                    current_upload is None
                    or current_upload.get("name") != uploaded_file.name
                    or current_upload.get("bytes") != file_bytes
                ):
                    st.session_state["uploaded_csv"] = {"name": uploaded_file.name, "bytes": file_bytes}
                    reset_data_flow_state()
                    st.rerun()

    source_left, source_right = st.columns([3, 1])
    source_left.caption(f"{data_source_caption}: {data_source_name}")
    if uploaded_csv and source_right.button("기본 CSV로 되돌리기", key="reset_default_csv"):
        st.session_state.pop("uploaded_csv", None)
        reset_data_flow_state()
        st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.metric("행", f"{df.shape[0]:,}")
    c2.metric("컬럼", f"{df.shape[1]:,}")
    c3.metric("특성", f"{len(feature_cols):,}")
    st.markdown("### 컬럼 정보")
    st.dataframe(pd.DataFrame({"컬럼명": df.columns, "컬럼한글명": [kor_col(c) for c in df.columns], "데이터 타입": [str(t) for t in df.dtypes], "결측치 수": df.isna().sum().values}), width='stretch', height=280)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("### 범주형 컬럼")
    c1.dataframe(pd.DataFrame({"컬럼명": categorical_cols, "컬럼한글명": [kor_col(c) for c in categorical_cols]}), width='stretch', height=220)
    c2.markdown("### 수치형 컬럼")
    c2.dataframe(pd.DataFrame({"컬럼명": numeric_cols, "컬럼한글명": [kor_col(c) for c in numeric_cols]}), width='stretch', height=220)
    
    st.markdown("### 데이터 미리보기 (100개)")
    st.dataframe(df.head(100), width='stretch', height=420)
### 2. EDA ###
with tabs[1]:
    st.markdown("### EDA(탐색적 데이터 분석)")
    st.caption("데이터의 규모, 수치형 변수 분포, 범주형 변수 분포, 이탈과의 관계를 순서대로 확인합니다.")

    e1, e2, e3 = st.columns(3)
    e1.metric("전체 고객", f"{len(df):,}")
    e2.metric("수치형 컬럼", f"{len(numeric_cols):,}")
    e3.metric("범주형 컬럼", f"{len(categorical_cols):,}")

    st.markdown("#### 1. 이탈 현황")
    left, right = st.columns([1, 1])
    with left:
        fig, ax = compact_fig(4.8, 3.0)
        y.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#22c55e", "#ef4444"])
        ax.set_title("이탈 여부별 고객 수", fontsize=11)
        ax.set_xlabel("이탈 여부(0 : 유지, 1 : 이탈)", fontsize=9)
        ax.set_ylabel("고객 수", fontsize=9)
        ax.tick_params(axis="x", rotation=0)
        style_axis(ax)
        st.pyplot(fig, width='content')
    with right:
        churn_summary = pd.DataFrame({
            "구분": ["유지 고객", "이탈 고객"],
            "고객 수": [int((y == 0).sum()), int((y == 1).sum())],
            "비율": [f"{(y == 0).mean() * 100:.1f}%", f"{(y == 1).mean() * 100:.1f}%"],
        })
        st.dataframe(churn_summary, width='stretch', hide_index=True)

    st.markdown("#### 2. 특성별 분포")
    left, right = st.columns([1, 1])
    with left:
        if categorical_cols:
            cat = st.selectbox("범주형 특성 선택", categorical_cols, key="eda_cat", format_func=display_col)
            fig, ax = compact_fig(5.0, 3.0)
            df[cat].value_counts().head(8).plot(kind="bar", ax=ax, color="#14b8a6")
            ax.set_title(f"{kor_col(cat)} 항목별 고객 수", fontsize=11)
            ax.set_xlabel(kor_col(cat), fontsize=9)
            ax.set_ylabel("고객 수", fontsize=9)
            ax.tick_params(axis="x", rotation=25)
            style_axis(ax)
            st.pyplot(fig, width='content')
    with right:
        if numeric_cols:
            col = st.selectbox("수치형 특성 선택", numeric_cols, key="eda_num", format_func=display_col)
            fig, ax = compact_fig(5.0, 3.0)
            ax.hist(df[col].dropna(), bins=24, color="#2563eb", alpha=0.85, edgecolor="#ffffff", linewidth=1.4)
            ax.set_title(f"{kor_col(col)} 분포", fontsize=11)
            ax.set_xlabel(kor_col(col), fontsize=9)
            ax.set_ylabel("고객 수", fontsize=9)
            style_axis(ax)
            st.pyplot(fig, width='content')


    # st.markdown("#### 3. 이탈과 특성 간의 관계 분석")
    # left, right = st.columns([1, 1])

    # # # 시각화용 임시 리스트 생성: numeric_cols에서 'underutilized_premium'만 제외
    # # plot_cols = [col for col in numeric_cols if col != 'underutilized_premium']

    # with left:
    #     st.markdown("##### (1) 이탈과 상관관계가 큰 수치형 특성")
    #     # numeric_cols 대신 plot_cols를 사용합니다.
    #     corr_df = df[plot_cols + [target_col]].corr(numeric_only=True)[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        
    #     # 가로 폭을 줄여 컴팩트하게 설정 (예: 4.5 x 3.5)
    #     fig_bar, ax_bar = compact_fig(4.5, 3.5)
    #     corr_df.head(8).sort_values().plot(kind="barh", ax=ax_bar, color="#f97316")
    #     ax_bar.set_title("이탈과 상관관계 지표", fontsize=10)
    #     ax_bar.set_xlabel("상관계수", fontsize=8)
    #     style_axis(ax_bar)
        
    #     st.pyplot(fig_bar, width='stretch')    # 컬럼너비맞춤

    # with right:
    #     st.markdown("##### (2) 수치형 특성 간 상관관계 히트맵")
    #     # 마찬가지로 numeric_cols 대신 plot_cols를 사용합니다.
    #     corr_matrix = df[plot_cols + [target_col]].corr(numeric_only=True)
        
    #     # 바 차트와 일관된 크기로 설정 (4.5 x 3.5)
    #     fig_hm, ax_hm = compact_fig(4.5, 3.5)
        
    #     sns.heatmap(
    #         corr_matrix, 
    #         annot=True, 
    #         fmt=".2f", 
    #         cmap="coolwarm", 
    #         vmin=-1, vmax=1, 
    #         ax=ax_hm, 
    #         annot_kws={"size": 6},  # 가로 폭이 좁아지므로 글자 크기를 더 축소
    #         cbar=False,             # 양 옆 균형을 위해 컬러바 제거 (선택 사항)
    #         linewidths=0.5
    #     )
    #     ax_hm.set_title("특성 간 상관관계 매트릭스", fontsize=10)
        
    #     # 좁은 공간에서 글자가 겹치지 않도록 세밀하게 조절
    #     ax_hm.tick_params(axis='x', labelsize=7, rotation=45)
    #     ax_hm.tick_params(axis='y', labelsize=7, rotation=0)
    #     style_axis(ax_hm)
        
    #     # col2 영역에 그래프 출력
    #     st.pyplot(fig_hm, width='stretch')


    # st.markdown("#### ")
    # left, right = st.columns([1, 1])

    # with left:
    #     st.markdown("##### (3) 총 청취시간에 따른 분포")
    #     fig_kde, ax_kde = compact_fig(4.5, 3.5)
        
    #     sns.kdeplot(
    #         data=df,
    #         x="listening_time",
    #         hue="Churn",
    #         fill=True,
    #         common_norm=False,
    #         palette="crest",
    #         alpha=0.5,
    #         ax=ax_kde
    #     )
        
    #     ax_kde.set_title("총 청취 시간에 따른 이탈/유지 분포 (KDE)", fontsize=10)
    #     ax_kde.set_xlabel("총 청취 시간 (분)", fontsize=8)
    #     ax_kde.set_ylabel("밀도 (Density)", fontsize=8)

    #     style_axis(ax_kde)
        
    #     st.pyplot(fig_kde, width='stretch')

    # with right:
    #     st.markdown("##### (4) 한 곡당 평균 청취 시간에 따른 분포")
    #     fig_kde, ax_kde = compact_fig(4.5, 3.5)
        
    #     sns.kdeplot(
    #         data=df,
    #         x="avg_time_per_song",
    #         hue="is_churned",
    #         fill=True,
    #         common_norm=False,
    #         palette="crest",
    #         alpha=0.5,
    #         ax=ax_kde
    #     )
        
    #     ax_kde.set_title("한 곡당 평균 청취 시간에 따른 이탈/유지 분포 (KDE)", fontsize=10)
    #     ax_kde.set_xlabel("한 곡당 평균 청취 시간 (분)", fontsize=8)
    #     ax_kde.set_ylabel("밀도 (Density)", fontsize=8)

    #     style_axis(ax_kde)
        
    #     st.pyplot(fig_kde, width='stretch')

    # st.markdown("#### ")
    # left, right = st.columns([1, 1])

    # with left:
    #     st.markdown("##### (5) 구독 요금제별 분포")
    #     # fig_kde, ax_kde = compact_fig(4.5, 3.5)

    #     sns.countplot(
    #         data=df,
    #         x="subscription_type",
    #         hue="is_churned",
    #         palette="pastel",
    #         ax=ax_kde
    #     )
        
    #     ax_kde.set_title("구독 요금제 유형별 이탈/유지 유저 수", fontsize=10)
    #     ax_kde.set_xlabel("구독 요금제 유형", fontsize=8)
    #     ax_kde.set_ylabel("유저 수", fontsize=8)

    #     style_axis(ax_kde)
        
    #     st.pyplot(fig_kde, width='stretch')
        

    # with right:
### 3. 전처리 ###
with tabs[2]:
    st.markdown("### 전처리 설계")
    current_test_size = st.session_state.get("test_size", 0.2)
    current_cv_n = st.session_state.get("cv_n", 5)
    train_pct = int(round((1 - current_test_size) * 100))
    test_pct = int(round(current_test_size * 100))
    st.markdown(f"""
    <div class='glass'>
    <b>적용 내용</b><br>
    - user_id 컬럼은 학습에서 제외<br>
    - 범주형 특성: OneHotEncoder 적용<br>
    - 수치형 특성: StandardScaler 적용<br>
    </div>
    """, unsafe_allow_html=True)
### 4. 모델링 ###
with tabs[3]:
    st.markdown("### 모델 학습")
    st.caption("모델을 선택하면 최적 하이퍼파라미터가 자동으로 탐색되고, 탐색 결과가 모델 학습 설정에 반영됩니다.")

    model_options = ["모델을 선택하세요", "Logistic Regression", "SGD Classifier", "KNN", "SVM", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"]
    current_model = st.session_state.get("model_name", "모델을 선택하세요")
    model_search_done = (
        current_model != "모델을 선택하세요"
        and st.session_state.get("best_model_name") == current_model
        and "best_params" in st.session_state
    )
    step_title("1. 모델 선택", model_search_done)
    model_name = st.selectbox("모델 선택", model_options, index=0, key="model_name")
    previous_flow_model = st.session_state.get("model_flow_model_name")
    if previous_flow_model != model_name:
        reset_model_flow_state()
        st.session_state["model_flow_model_name"] = model_name
        if previous_flow_model is not None:
            st.rerun()

    grid_map = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 3.0, 10.0]},
        "SGD Classifier": {"model__alpha": [0.0001, 0.001, 0.01, 0.1]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 11, 15]},
        "SVM": {"model__C": [0.1, 1.0, 3.0], "model__kernel": ["rbf", "linear"]},
        "Decision Tree": {"model__max_depth": [3, 5, 7, 10, 15], "model__min_samples_split": [2, 5, 10]},
        "Random Forest": {"model__n_estimators": [50, 100, 150], "model__max_depth": [3, 5, 8, 12], "model__min_samples_split": [2, 5]},
        "Gradient Boosting": {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.05, 0.1, 0.2], "model__max_depth": [2, 3, 5]},
        "AdaBoost": {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.05, 0.1, 0.3]},
    }
    base_defaults = {"C": 1.0, "kernel": "rbf", "alpha": 0.001, "n_neighbors": 5, "max_depth": 5, "min_samples_split": 2, "n_estimators": 100, "learning_rate": 0.1}
    if model_name == "모델을 선택하세요":
        pass
    else:
        has_grid_result = "best_params" in st.session_state and st.session_state.get("best_model_name") == model_name
        if not has_grid_result:
            with st.spinner("선택한 모델의 최적 하이퍼파라미터를 탐색하는 중입니다..."):
                pre = make_preprocessor(numeric_cols, categorical_cols)
                estimator = build_model(model_name, base_defaults, svm_probability=False)
                tune_pipe = Pipeline([("preprocess", pre), ("model", estimator)])
                grid = GridSearchCV(tune_pipe, grid_map[model_name], cv=3, scoring="f1", n_jobs=-1)
                grid.fit(X, y)
                st.session_state["best_params"] = grid.best_params_
                st.session_state["best_score"] = grid.best_score_
                st.session_state["best_model_name"] = model_name
                st.session_state["cv_results"] = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").head(10)
                st.session_state["pending_manual_params"] = grid.best_params_
            st.rerun()

        if has_grid_result:
            show_best_params_box(
                st.session_state["best_model_name"],
                st.session_state["best_params"],
                float(st.session_state["best_score"]),
            )
            substep_title("GridSearchCV 후보별 성능 순위")
            top_results = st.session_state["cv_results"][["rank_test_score", "params", "mean_test_score", "std_test_score"]].copy()
            top_results["params"] = top_results["params"].apply(format_param_dict)
            top_results = top_results.rename(
                columns={
                    "rank_test_score": "순위",
                    "mean_test_score": "교차검증 F1 점수",
                    "std_test_score": "교차검증 F1 표준편차",
                    "params": "하이퍼파라미터 조합",
                }
            )
            top_results = top_results[["순위", "하이퍼파라미터 조합", "교차검증 F1 점수", "교차검증 F1 표준편차"]]
            st.dataframe(top_results, width='stretch', height=280, hide_index=True)

            st.divider()
            user_train_done = "train_metrics" in st.session_state and st.session_state.get("trained_model_name") == model_name
            step_title("2. 모델 학습", user_train_done)
            substep_title("2-1. 데이터셋 검증 설정")
            c1, c2 = st.columns(2)
            with c1:
                test_size = st.slider("test_size", 0.1, 0.4, 0.2, 0.05, key="test_size")
            with c2:
                cv_n = st.slider("StratifiedKFold 수", 3, 10, 5, key="cv_n")
            substep_title("2-2. 하이퍼파라미터 설정")
            params = params_ui(model_name, "manual")

            # app.py 내부 Tab 4 '모델 학습 실행' 버튼 클릭 시점
            if st.button("모델 학습 실행", type="primary", key="manual_train_button"):
                
                # 1. 수정한 모듈로부터 불균형이 해소된(Balanced) 트레인 데이터를 받아옵니다.
                X_train_balanced, X_test_trans, y_train_balanced, y_test, trained_preprocessor = preprocessing_util.preprocess_data(
                    df, test_size=test_size, random_state=RANDOM_STATE
                )
                
                # 2. 선택한 모델 빌드
                model = build_model(model_name, params)
                
                # 3. [주의] 파이프라인으로 묶어서 fit을 하면 전처리가 중복으로 실행되므로, 
                # 전처리가 이미 완료된 X_train_balanced로 모델만 따로 학습(fit)시킵니다.
                model.fit(X_train_balanced, y_train_balanced)
                
                # 4. 평가를 위해 전체 프로세스를 파이프라인 구조로 묶어 세션에 저장합니다.
                pipe = Pipeline([("preprocess", trained_preprocessor), ("model", model)])
                
                # 5. 예측 진행 (테스트 데이터 원본을 넣어 파이프라인을 통과시킵니다)
                _, X_test_raw, _, _ = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
                pred = pipe.predict(X_test_raw)
                proba = predict_probability(pipe, X_test_raw)
                
                # 교차 검증(CV) 설정
                cv = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=RANDOM_STATE)
                cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

                # 세션 상태 저장 (기존 코드 유지)
                st.session_state["pipe"] = pipe
                st.session_state["trained_model_name"] = model_name
                st.session_state["params"] = params
                st.session_state["X_test"] = X_test_raw
                st.session_state["y_test"] = y_test
                st.session_state["pred"] = pred
                st.session_state["proba"] = proba
                st.session_state["numeric_cols"] = numeric_cols
                st.session_state["categorical_cols"] = categorical_cols
                st.session_state["train_metrics"] = {
                    "accuracy": accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "cv_f1": cv_scores.mean(),
                }
                st.session_state["train_settings"] = {
                    "model": model_name,
                    "manual_params": params,
                    "test_size": test_size,
                    "cv": f"StratifiedKFold({cv_n})",
                    "numeric_scaling": "StandardScaler + SMOTE 밸런싱 적용",
                }
                st.rerun()

            if user_train_done:
                # step_title("3. 모델 학습 결과", user_train_done)
                # metrics = st.session_state["train_metrics"]
                # m1, m2, m3, m4, m5 = st.columns(5)
                # m1.metric("정확도", f"{metrics['accuracy']:.3f}")
                # m2.metric("정밀도", f"{metrics['precision']:.3f}")
                # m3.metric("재현율", f"{metrics['recall']:.3f}")
                # m4.metric("F1 점수", f"{metrics['f1']:.3f}")
                # m5.metric("교차검증 F1 점수", f"{metrics['cv_f1']:.3f}")
                substep_title("모델 학습 설정")
                st.json(st.session_state["train_settings"])
            else:
                st.info("모델 학습 실행 버튼을 누르면 평가 지표와 모델 해석, Spotify 고객 이탈 예측 탭이 활성화됩니다.")

with tabs[4]:
    st.markdown("### 평가 지표")
    if "pipe" not in st.session_state:
        st.info("4. 모델 학습 탭에서 모델 학습 실행 버튼을 눌러야 평가 지표 탭이 활성화됩니다.")
    else:
        metrics = st.session_state["train_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("정확도", f"{metrics['accuracy']:.3f}")
        m2.metric("정밀도", f"{metrics['precision']:.3f}")
        m3.metric("재현율", f"{metrics['recall']:.3f}")
        m4.metric("F1 점수", f"{metrics['f1']:.3f}")
        m5.metric("교차검증 F1 점수", f"{metrics['cv_f1']:.3f}")

        y_test = st.session_state["y_test"]
        pred = st.session_state["pred"]
        proba = st.session_state["proba"]
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_test, pred)
            fig, ax = compact_fig(3.6, 3.0)
            ax.imshow(cm)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
            ax.set_title("Confusion Marix", fontsize=10)
            ax.set_xlabel("예측값", fontsize=9)
            ax.set_ylabel("실제값", fontsize=9)
            st.pyplot(fig, width='content')
        with c2:
            fig, ax = compact_fig(4.4, 3.0)
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title("ROC Curve", fontsize=10)
            ax.legend(fontsize=8)
            style_axis(ax)
            st.pyplot(fig, width='content')
        report_df = pd.DataFrame(classification_report(y_test, pred, zero_division=0, output_dict=True)).T
        report_df = report_df.rename(
            columns={
                "precision": "정밀도",
                "recall": "재현율",
                "f1-score": "F1 점수",
                "support": "표본 수",
            }
        )
        st.markdown("#### 분류 리포트")
        st.dataframe(report_df.round(3), width='stretch')

with tabs[5]:
    st.markdown("### 모델 해석")
    if "pipe" not in st.session_state:
        st.info("4. 모델 학습 탭에서 모델 학습 실행 버튼을 눌러야 모델 해석 탭이 활성화됩니다.")
    else:
        pipe = st.session_state["pipe"]
        model = pipe.named_steps["model"]
        names = get_feature_names(pipe, numeric_cols, categorical_cols)
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).ravel()
        else:
            result = permutation_importance(pipe, st.session_state["X_test"], st.session_state["y_test"], n_repeats=5, random_state=RANDOM_STATE, scoring="f1")
            imp = result.importances_mean
            names = feature_cols
        imp_df = pd.DataFrame({"feature": [kor_col(n) for n in names], "importance": imp, "original_feature": names}).sort_values("importance", ascending=False).head(12)
        fig, ax = compact_fig(5.6, 3.4)
        imp_df.sort_values("importance").plot(kind="barh", x="feature", y="importance", ax=ax, legend=False)
        ax.set_title("특성 수 Importance", fontsize=10)
        style_axis(ax)
        st.pyplot(fig, width='content')
        st.dataframe(imp_df, width='stretch', height=280)

with tabs[6]:
    st.markdown("### 모델 설명")
    if "pipe" not in st.session_state:
        st.info("4. 모델 학습 탭에서 모델 학습 실행 버튼을 눌러야 Spotify 고객 이탈 예측 탭이 활성화됩니다.")
    else:
        # st.markdown("#### 현재 예측에 사용되는 모델과 하이퍼파라미터")
        st.markdown(
            f"""
            <div class='glass'>
              <b>사용 모델</b><br>
              <span style='font-size:1.25rem; font-weight:800;'>{st.session_state["trained_model_name"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        if st.session_state["params"]:
            param_df = pd.DataFrame({
                "하이퍼파라미터": list(st.session_state["params"].keys()),
                "설정값": list(st.session_state["params"].values()),
            })
            st.dataframe(param_df, width='stretch', hide_index=True)
        else:
            st.info("현재 모델에는 표시할 하이퍼파라미터가 없습니다.")


        st.markdown("### Spotify 고객 이탈 예측")
        values = {}
        cols = st.columns(2)
        idx = 0
        for col in categorical_cols:
            opts = sorted(df[col].dropna().astype(str).unique().tolist())
            with cols[idx % 2]:
                values[col] = st.selectbox(display_col(col), opts, key=f"pred_{col}")
            idx += 1
        for col in numeric_cols:
                    is_int_col = is_integer_series(df[col])
                    with cols[idx % 2]:
                        if is_int_col:
                            min_v = int(df[col].min())
                            max_v = int(df[col].max())
                            mean_v = int(round(float(df[col].mean())))
                            
                            # [추가] 최솟값과 최댓값이 같을 때 (예: 0과 0) 에러 방지 예외 처리
                            if min_v == max_v:
                                # 0 또는 1만 가질 수 있는 특정 파생변수('underutilized_premium' 등)를 위해 라디오 버튼이나 고정값 제공
                                if col == 'underutilized_premium' or min_v in [0, 1]:
                                    # 0과 1 중에서 선택할 수 있도록 옵션을 강제로 [0, 1]로 제공하는 라디오 버튼 생성
                                    choice = st.radio(display_col(col), options=[0, 1], index=min_v, key=f"pred_{col}")
                                    values[col] = choice
                                else:
                                    # 그 외 완전히 값이 고정된 컬럼은 단순 안내창 및 고정값 할당
                                    st.caption(f"{display_col(col)} (고정값: {min_v})")
                                    values[col] = min_v
                            else:
                                # 최솟값과 최댓값이 다를 때만 정상적으로 슬라이더 생성
                                values[col] = st.slider(display_col(col), min_v, max_v, mean_v, step=1, key=f"pred_{col}")
                        else:
                            min_v = float(df[col].min())
                            max_v = float(df[col].max())
                            mean_v = float(df[col].mean())
                            
                            # [추가] 수치형(실수형) 컬럼도 동일하게 예외 처리 적용
                            if min_v == max_v:
                                st.caption(f"{display_col(col)} (고정값: {min_v:.2f})")
                                values[col] = min_v
                            else:
                                values[col] = st.slider(display_col(col), min_v, max_v, mean_v, key=f"pred_{col}")
                    idx += 1
                    
        input_df = pd.DataFrame([values])[feature_cols]
        if st.button("이탈 예측 실행", type="primary"):
            pipe = st.session_state["pipe"]
            pred_label = int(pipe.predict(input_df)[0])
            prob = float(predict_probability(pipe, input_df)[0])
            st.info(f"예측 사용 모델: {st.session_state['trained_model_name']}")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("이탈 가능성", f"{prob*100:.1f}%")
            with c2:
                st.metric("예측 결과", "이탈 가능성 높음" if pred_label == 1 else "유지 가능성 높음")
            if prob >= 0.6:
                st.markdown("<div class='risk-high'><b>위험도 높음</b><br>이 사용자는 이탈 가능성이 높게 예측되었습니다.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-low'><b>위험도 낮음</b><br>현재 입력값 기준 이탈 가능성은 낮게 예측되었습니다.</div>", unsafe_allow_html=True)
            st.markdown("#### 어떤 부분이 문제인가")
            risk_notes = []
            for col in numeric_cols:
                val = float(input_df[col].iloc[0])
                churn_mean = float(df.loc[y == 1, col].mean()) if (y == 1).any() else float(df[col].mean())
                non_mean = float(df.loc[y == 0, col].mean()) if (y == 0).any() else float(df[col].mean())
                total_std = float(df[col].std()) if float(df[col].std()) > 0 else 1.0
                if abs(val - churn_mean) < abs(val - non_mean) and abs(churn_mean - non_mean) > 0.15 * total_std:
                    direction = "높은" if churn_mean > non_mean else "낮은"
                    risk_notes.append(f"- `{kor_col(col)}` 값이 이탈 고객 평균에 더 가깝습니다. 이탈 고객은 보통 `{kor_col(col)}`이 더 {direction} 편입니다.")
            for col in categorical_cols:
                val = input_df[col].iloc[0]
                rate_by_cat = df.assign(_y=y).groupby(col)["_y"].mean()
                if val in rate_by_cat.index and rate_by_cat.loc[val] > y.mean() + 0.05:
                    risk_notes.append(f"- `{kor_col(col)} = {val}` 그룹의 평균 이탈률이 전체 평균보다 높습니다.")
            if not risk_notes:
                risk_notes.append("- 현재 입력값에서 뚜렷하게 튀는 위험 요인은 크지 않습니다. 모델의 복합 패턴으로 예측된 결과입니다.")
            st.markdown("\n".join(risk_notes[:6]))
