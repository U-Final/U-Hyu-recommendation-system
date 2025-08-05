from collections import defaultdict
import pandas as pd
import random

'''
user, brand, interactions feature 구성
'''

def build_user_features(user_brand_df, bookmark_df, brand_df, exclude_brand_ids=None):
    exclude_brand_ids = set(exclude_brand_ids or [])

    user_feature_map = defaultdict(list)
    bookmark_map = bookmark_df.groupby("user_id")["brand_id"].apply(list).to_dict()
    brand_to_category = dict(zip(brand_df["brand_id"], brand_df["category_id"]))

    # 최종적으로 부여되는 가중치 비중은 (최대로 데이터를 가져왔을 때) 최대 개수 * 가중치
    # 관심 브랜드 > 방문 브랜드 > 즐겨찾기 블랜드
    # 사용자마다 데이터 수의 편차가 있을 것이고, 아직은 초창기라 데이터가 많지 않을 것을 고려하여 다음과 같이 개수 지정
    # 실사용자 받아서 몇개의 관심 브랜드 데이터 / 방문 브랜드 데이터 / 북마크 데이터가 몇개정도 저장하는지 확인해서 지정 예정
    for user_id, group in user_brand_df.groupby("user_id"):
        interest = group[(group["data_type"] == "INTEREST") & (~group["brand_id"].isin(exclude_brand_ids))]["brand_id"].tolist()[:5]
        recent = group[(group["data_type"] == "RECENT") & (~group["brand_id"].isin(exclude_brand_ids))]["brand_id"].tolist()[:5]
        bookmarked = [b for b in bookmark_map.get(user_id, []) if b not in exclude_brand_ids][:5]

        features = (
            [f"interest_{b}" for b in interest] * 3 +
            [f"recent_{b}" for b in recent] * 2 +
            [f"bookmark_{b}" for b in bookmarked] * 1
        )

        category_ids = {brand_to_category.get(b) for b in recent + interest + bookmarked if brand_to_category.get(b)}
        features += [f"cat_{cid}" for cid in category_ids] * 2

        user_feature_map[user_id] = features

    return user_feature_map

def build_item_features(brand_df):
    item_feature_map = {}

    for _, row in brand_df.iterrows():
        brand_id = row["brand_id"]
        features = []

        # 카테고리
        category_id = row.get("category_id")
        if not pd.isna(category_id) and pd.api.types.is_numeric_dtype(type(category_id)):
            try:
                features.append(f"category_{int(category_id)}")
            except (ValueError, TypeError):
                pass

        # 온라인/오프라인
        store_type = row.get("store_type")
        if not pd.isna(store_type) and isinstance(store_type, str):
            features.append(f"store_{store_type.lower()}")

        # 예시: 브랜드명 키워드 (간단 토크나이징)
        brand_name = row.get("brand_name")
        if not pd.isna(brand_name) and isinstance(brand_name, str):
            tokens = brand_name.lower().split()
            features += [f"name_{token}" for token in tokens]

        item_feature_map[brand_id] = features

    return item_feature_map

def build_interactions(dataset, interaction_df, user_brand_df, brand_df):
    users_with_logs = set(interaction_df["user_id"])
    all_users = set(user_brand_df["user_id"].unique())
    users_without_logs = all_users - users_with_logs

    dummy_interactions = []
    for user_id in users_without_logs:
        group = user_brand_df[user_brand_df["user_id"] == user_id]
        recent = group[group["data_type"] == "RECENT"]["brand_id"].tolist()
        interest = group[group["data_type"] == "INTEREST"]["brand_id"].tolist()

        dummy_interactions += [(user_id, b, 2.0) for b in interest]
        dummy_interactions += [(user_id, b, 3.0) for b in recent]

        if not interest and not recent:
            random.seed(user_id)
            random_brand = random.choice(brand_df["brand_id"].tolist())
            dummy_interactions.append((user_id, random_brand, 1.0))

    real = list(zip(interaction_df["user_id"], interaction_df["brand_id"], interaction_df["weight"]))
    return dataset.build_interactions(real + dummy_interactions)