from collections import defaultdict
import pandas as pd

'''
LightFM 행렬 구성
'''

def build_user_features(user_brand_df, bookmark_df, brand_df, exclude_brand_ids=None):
    exclude_brand_ids = set(exclude_brand_ids or [])

    user_feature_map = defaultdict(list)
    bookmark_map = bookmark_df.groupby("user_id")["brand_id"].apply(list).to_dict()
    brand_to_category = dict(zip(brand_df["brand_id"], brand_df["category_id"]))

    for user_id, group in user_brand_df.groupby("user_id"):
        recent = group[(group["data_type"] == "RECENT") & (~group["brand_id"].isin(exclude_brand_ids))]["brand_id"].tolist()[:6]
        interest = group[(group["data_type"] == "INTEREST") & (~group["brand_id"].isin(exclude_brand_ids))]["brand_id"].tolist()[:3]
        bookmarked = [b for b in bookmark_map.get(user_id, []) if b not in exclude_brand_ids][:5]

        features = (
            [f"recent_{b}" for b in recent] * 2 +
            [f"interest_{b}" for b in interest] * 3 +
            [f"bookmark_{b}" for b in bookmarked] * 2
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