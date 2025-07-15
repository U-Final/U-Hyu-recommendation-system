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
print("📥 사용자, 브랜드, 상호작용 데이터 로딩 중...")

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
        SELECT al.user_id, b.brand_id, al.action_type
        FROM action_logs al
        JOIN stores s ON al.store_id = s.id
        JOIN brands b ON s.brand_id = b.brand_id
        WHERE al.action_type IN ('marker_click', 'favorite')
        """
    )).fetchall(), columns=["user_id", "brand_id", "action_type"])

    # 즐겨찾기 목록

    #

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

(interactions, weights) = dataset.build_interactions([
    (row["user_id"], row["brand_id"], row["weight"])
    for _, row in interaction_df.iterrows()
])

user_features = dataset.build_user_features(
    [(uid, feats) for uid, feats in user_feature_map.items()]
)


# 5. 모델 학습
print("🧠 LightFM 모델 학습 중...")
model = LightFM(loss="warp")
model.fit(interactions, user_features=user_features, sample_weight=weights, epochs=10, num_threads=2)

# 6. 추천 생성
print("📊 사용자별 추천 생성 중...")

all_item_ids = brand_df["brand_id"].tolist()
recommendations = []

for user_id in user_df["user_id"]:
    known = interaction_df[interaction_df["user_id"] == user_id]["brand_id"].tolist()

    scores = model.predict(user_ids=user_id, item_ids=np.array(all_item_ids))
    scores = [(item_id, s) for item_id, s in zip(all_item_ids, scores) if item_id not in known]
    top_k = sorted(scores, key=lambda x: -x[1])[:5]

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
# print("💾 추천 결과 DB 저장 중...")
#
# with engine.begin() as conn:
#     user_ids = recommend_df["user_id"].unique().tolist()
#
#     # 해당 사용자들의 기존 추천 삭제
#     conn.execute(text(
#         "DELETE FROM recommendation WHERE user_id = ANY(:uids)"
#     ), {"uids": user_ids})
#
#     # 삽입 반복
#     for _, row in recommend_df.iterrows():
#         conn.execute(text("""
#             INSERT INTO recommendation (user_id, brand_id, score, rank, created_at)
#             VALUES (:user_id, :brand_id, :score, :rank, :created_at)
#         """), {
#             "user_id": int(row.user_id),
#             "brand_id": int(row.brand_id),
#             "score": float(row.score),
#             "rank": int(row.rank),
#             "created_at": row.created_at
#         })
#
# print("✅ 추천 완료 및 DB 저장 완료.")