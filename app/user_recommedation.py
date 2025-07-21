import os
import pandas as pd
from sqlalchemy import create_engine, text
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 불러오기
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# DB 연결 URL 생성
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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
    interaction_df = pd.DataFrame(conn.execute(text(
        """
        SELECT al.user_id, b.id, al.action_type
        FROM action_logs al
        JOIN store s ON al.store_id = s.id
        JOIN brands b ON s.brand_id = b.id
        WHERE al.action_type IN ('MARKER_CLICK', 'FILTER_USED')
        """
    )).fetchall(), columns=["user_id", "brand_id", "action_type"])

    # 즐겨찾기 목록 정보

# 4. 온보딩 기반 user_feature 구성
user_feature_map = defaultdict(list)

for user_id, group in onboarding_df.groupby("user_id"):
    recent = group[group["data_type"] == "RECENT"]["brand_id"].tolist()[:6]
    interest = group[group["data_type"] == "INTEREST"]["brand_id"].tolist()[:4]

    # 브랜드 아이디에 prefix를 붙여서 feature로 만듦
    features = [f"recent_{b}" for b in recent] + [f"interest_{b}" for b in interest]
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

# 실제 interaction은 action_logs 기반으로 구성
real_interactions = list(zip(interaction_df["user_id"], interaction_df["brand_id"]))

# action log가 없는 유저는 첫 브랜드만 대상으로 dummy interaction 생성
dummy_interactions = [(user_id, brand_df["brand_id"].iloc[0]) for user_id in users_without_logs]

combined_interactions = real_interactions + dummy_interactions

interactions, _ = dataset.build_interactions(combined_interactions)

user_features = dataset.build_user_features(
    [(uid, feats) for uid, feats in user_feature_map.items()]
)

# 모델 학습
print("🧠 LightFM 모델 학습 중...")
model = LightFM(loss="warp")
model.fit(interactions, user_features=user_features, epochs=10, num_threads=2)

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

# 7. 추천 결과 CSV로 저장
print("💾 추천 결과 CSV 저장 중...")

csv_path = "recommendations.csv"
recommend_df.to_csv(csv_path, index=False)

print(f"✅ 추천 완료 및 CSV 저장 완료: {csv_path}")

# 7. 추천 결과 저장 (SQLAlchemy Core 사용)
print("💾 추천 결과 DB 저장 중...")

with engine.begin() as conn:
    for _, row in recommend_df.iterrows():
        conn.execute(text("""
            INSERT INTO recommendation (user_id, brand_id, score, rank, created_at)
            VALUES (:user_id, :brand_id, :score, :rank, :created_at)
        """), {
            "user_id": int(row.user_id),
            "brand_id": int(row.brand_id),
            "score": float(row.score),
            "rank": int(row['rank']),
            "created_at": row.created_at
        })

print("✅ 추천 완료 및 DB 저장 완료.")