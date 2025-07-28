import os
import pandas as pd
# 추천 결과와 클릭 행동 간의 카테고리 분포를 시각화하는 코드입니다.
# 필요 라이브러리인 matplotlib과 seaborn을 import합니다.
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 필수 환경 변수 검증
required_env_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# 1. DB 연결 (환경 변수에서 정보 읽기)
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

# 2. 데이터 로딩
print("📥 사용자, 브랜드, interaction 데이터 로딩 중...")

with engine.connect() as conn:
    # 유저 정보
    user_df = pd.DataFrame(conn.execute(text(
        "SELECT id, gender, age_range FROM users"
    )).fetchall(), columns=["user_id", "gender", "age_range"])

    # 온보딩 정보
    onboarding_df = pd.DataFrame(conn.execute(text(
        """
        SELECT user_id, brand_id, data_type
        FROM recommendation_base_data
        """
    )).fetchall(), columns=["user_id", "brand_id", "data_type"])

    # 브랜드 & 카테고리 정보
    brand_df = pd.DataFrame(conn.execute(text(
        "SELECT id, brand_name, category_id FROM brands"
    )).fetchall(), columns=["brand_id", "brand_name", "category_id"])

    # 행동 로그 정보
    interaction_raw = pd.DataFrame(conn.execute(text(
        """
        SELECT al.user_id, b.id AS brand_id, al.action_type
        FROM action_logs al
        JOIN store s ON al.store_id = s.id
        JOIN brands b ON s.brand_id = b.id
        WHERE al.action_type IN ('MARKER_CLICK', 'FILTER_USED')
        """
    )).fetchall(), columns=["user_id", "brand_id", "action_type"])

    # 행동 유형별로 중요도를 다르게 설정
    action_weights = {
        "MARKER_CLICK": 0.5,
        "FILTER_USED": 0.3
    }

    # 각 행동에 맞는 가중치를 weight 칼럼에 추가
    interaction_raw["weight"] = interaction_raw["action_type"].map(action_weights)
    # 같은 브랜드를 여러 번 클릭한 경우 가중치 누적 합산
    interaction_df = interaction_raw.groupby(["user_id", "brand_id"])["weight"].sum().reset_index()

    # 즐겨찾기 목록 정보

# 4. 온보딩 기반 user_feature 구성
user_feature_map = defaultdict(list)

for user_id, group in onboarding_df.groupby("user_id"):
    recent = group[group["data_type"] == "RECENT"]["brand_id"].tolist()[:6] # 방문 브랜드 최대 6개까지 추출
    interest = group[group["data_type"] == "INTEREST"]["brand_id"].tolist()[:4] # 관심 브랜드 최대 4개까지 추출

    # 브랜드 아이디에 prefix를 붙여서 feature로 만듦
    # recent 3배, interest 2배로 강조
    features = (
            [f"recent_{b}" for b in recent] * 3 +
            [f"interest_{b}" for b in interest] * 2
    )

    # category 정보도 함께 반영
    brand_to_category = dict(zip(brand_df["brand_id"], brand_df["category_id"]))
    category_ids = set()
    for b in recent + interest:
        category_id = brand_to_category.get(b)
        if category_id is not None:
            category_ids.add(category_id)

    features += [f"cat_{cid}" for cid in category_ids] * 2  # category 선호도 2배 강조
    user_feature_map[user_id] = features

# 5. LightFM 입력 구성
dataset = Dataset()
dataset.fit(users=user_df["user_id"], items=brand_df["brand_id"])

# 전체 유저 feature 집합
all_user_features = set(f for feats in user_feature_map.values() for f in feats)
dataset.fit_partial(user_features=all_user_features)

print("🧾 사용자별 interaction 구성 중...")

# Action log가 있는 유저와 없는 유저 분리
users_with_logs = set(interaction_df["user_id"])
all_users = set(user_df["user_id"])
users_without_logs = all_users - users_with_logs

# dummy_interactions 생성: 온보딩 기반, 가중치 반영
dummy_interactions = []
for user_id in users_without_logs:
    onboarding_rows = onboarding_df[onboarding_df["user_id"] == user_id]
    recent = onboarding_rows[onboarding_rows["data_type"] == "RECENT"]["brand_id"].tolist()
    interest = onboarding_rows[onboarding_rows["data_type"] == "INTEREST"]["brand_id"].tolist()

    # 관심 브랜드: 가중치 2, 방문 브랜드: 가중치 3
    dummy_interactions += [(user_id, b, 2.0) for b in interest]
    dummy_interactions += [(user_id, b, 3.0) for b in recent]

    # 아무 데이터도 없는 유저는 기본 브랜드 하나 넣기
    if not interest and not recent:
        dummy_interactions.append((user_id, brand_df["brand_id"].iloc[0], 1.0))

# real_interactions와 dummy_interactions 합쳐서 interactions 생성
real_interactions = list(zip(interaction_df["user_id"], interaction_df["brand_id"], interaction_df["weight"]))
all_interactions = real_interactions + dummy_interactions
interactions, weights = dataset.build_interactions(all_interactions)

user_features = dataset.build_user_features(
    [(uid, feats) for uid, feats in user_feature_map.items()]
)

# 모델 학습
print("🧠 LightFM 모델 학습 중...")
model = LightFM(loss="warp")
model.fit(interactions, sample_weight=weights, user_features=user_features, epochs=10, num_threads=2)

# 6. 추천 생성
print("📊 사용자별 추천 생성 중...")

all_item_ids = brand_df["brand_id"].tolist()
recommendations = []

for user_id in user_df["user_id"]:
#     scores = model.predict(
#         user_ids=[user_df[user_df["user_id"] == user_id].index[0]],
#         item_ids=np.arange(len(all_item_ids)),
#         user_features=user_features
#     )
    user_index = user_df[user_df["user_id"] == user_id].index[0]
    user_id_array = np.full(len(all_item_ids), user_index)

    scores = model.predict(
        user_ids=user_id_array,
        item_ids=np.arange(len(all_item_ids)),
        user_features=user_features
    )

    top_k_indices = np.argsort(-scores)[:5]
    top_k = [(all_item_ids[i], scores[i]) for i in top_k_indices]

    for rank, (brand_id, score) in enumerate(top_k, start=1):
        recommendations.append({
            "user_id": user_id,
            "brand_id": brand_id,
            "score": float(score),
            "rank": rank,
            "created_at": datetime.utcnow()
        })

recommend_df = pd.DataFrame(recommendations)

# 브랜드 카테고리 정보 추가 : 추천 결과에 카테고리 ID를 미리 병합
recommend_df = recommend_df.merge(
    brand_df[["brand_id", "category_id"]],
    on="brand_id",
    how="left"
)
recommend_df["updated_at"] = datetime.utcnow()

# 배치 INSERT
recommend_df.to_sql(
    "recommendation",
    engine,
    if_exists="append",
    index=False,
    method="multi"
)

# 7. 추천 결과 CSV로 저장
print("💾 추천 결과 CSV 저장 중...")

csv_path = "recommendations.csv"
recommend_df.to_csv(csv_path, index=False)

print(f"✅ 추천 완료 및 CSV 저장 완료: {csv_path}")

print("✅ 추천 완료 및 DB 저장 완료.")

# 최근 이용 브랜드와 추천 브랜드가 얼마나 겹치는지를 기반으로한 평가지표
def calculate_hit_rate(interaction_df, recommend_df, user_id):
    # 1. 최근 많이 클릭한 브랜드 Top-N (예: 5개)
    top_clicked = interaction_df[interaction_df["user_id"] == user_id] \
        .sort_values("weight", ascending=False).head(5)["brand_id"].tolist()

    # 2. 추천된 브랜드 Top-K
    recommended = recommend_df[recommend_df["user_id"] == user_id]["brand_id"].tolist()

    # 3. 교집합 확인
    hits = set(top_clicked) & set(recommended)
    return len(hits) / len(recommended) if recommended else 0

def plot_user_category_distribution(user_id, interaction_df, recommend_df, brand_df):
    # 클릭한 브랜드의 category 분포
    clicked_brands = interaction_df[interaction_df["user_id"] == user_id]["brand_id"]
    clicked_categories = brand_df[brand_df["brand_id"].isin(clicked_brands)]["category_id"].value_counts().sort_index()
    clicked_categories.name = "Clicked"

    # 추천 받은 브랜드의 category 분포
    recommended_brands = recommend_df[recommend_df["user_id"] == user_id]["brand_id"]
    recommended_categories = brand_df[brand_df["brand_id"].isin(recommended_brands)]["category_id"].value_counts().sort_index()
    recommended_categories.name = "Recommended"

    # 합치기
    category_df = pd.concat([clicked_categories, recommended_categories], axis=1).fillna(0)

    # 시각화
    category_df.plot(kind="bar", figsize=(10, 5))
    plt.title(f"User {user_id} - Category Distribution (Clicked vs Recommended)")
    plt.xlabel("Category ID")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 사용자 클릭 브랜드와 추천 브랜드 비교 함수 추가
def show_user_click_vs_recommendation(user_id, interaction_df, recommend_df, brand_df):
    # 사용자가 클릭한 브랜드 Top-N
    top_clicked = interaction_df[interaction_df["user_id"] == user_id] \
        .sort_values("weight", ascending=False).head(5)["brand_id"]

    top_clicked_brands = brand_df[brand_df["brand_id"].isin(top_clicked)][["brand_id", "brand_name"]]
    top_clicked_brands = top_clicked_brands.merge(interaction_df[interaction_df["user_id"] == user_id], on="brand_id")
    top_clicked_brands = top_clicked_brands[["brand_name", "weight"]].sort_values("weight", ascending=False)
    top_clicked_brands.columns = ["Clicked Brand", "Click Weight"]

    # 추천된 브랜드 Top-N
    recommended = recommend_df[recommend_df["user_id"] == user_id][["brand_id", "score", "rank"]]
    recommended = recommended.merge(brand_df, left_on="brand_id", right_on="brand_id")
    recommended = recommended[["brand_name", "score", "rank"]].sort_values("rank")
    recommended.columns = ["Recommended Brand", "Score", "Rank"]

    # 출력
    print("\n📌 사용자 클릭 브랜드 Top 5:")
    print(top_clicked_brands.to_string(index=False))

    print("\n🎯 추천된 브랜드 Top 5:")
    print(recommended.to_string(index=False))

    # 관심 브랜드
    interest_brands = onboarding_df[(onboarding_df["user_id"] == user_id) & (onboarding_df["data_type"] == "INTEREST")]["brand_id"]
    interest_brands_names = brand_df[brand_df["brand_id"].isin(interest_brands)]["brand_name"].tolist()

    # 방문 브랜드
    recent_brands = onboarding_df[(onboarding_df["user_id"] == user_id) & (onboarding_df["data_type"] == "RECENT")]["brand_id"]
    recent_brands_names = brand_df[brand_df["brand_id"].isin(recent_brands)]["brand_name"].tolist()

    print("\n⭐ 관심 브랜드 (INTEREST):")
    print(", ".join(interest_brands_names) if interest_brands_names else "없음")

    print("\n📍 방문 브랜드 (RECENT):")
    print(", ".join(recent_brands_names) if recent_brands_names else "없음")

# 예시: 사용자 ID 2번에 대해 시각화
# plot_user_category_distribution(user_id=2, interaction_df=interaction_df, recommend_df=recommend_df, brand_df=brand_df)

for i in range(1, 30) :
    print(f"user : {i}")
    show_user_click_vs_recommendation(user_id=i, interaction_df=interaction_df, recommend_df=recommend_df,
                                      brand_df=brand_df)
    print("\n==============\n")