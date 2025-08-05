from lightfm.data import Dataset
import random

'''
LightFM 모델 학습을 위한 데이터셋 구성, 상호작용 행렬 생성, 모델 학습을 수행하는 전체 파이프라인 함수들을 정의
'''

def prepare_dataset(user_df, brand_df, user_feature_map, item_feature_map=None):
    dataset = Dataset()

    user_ids = user_df["user_id"].tolist()
    item_ids = brand_df["brand_id"].tolist()

    # 전체 feature 목록 수집
    user_features = set(f for feats in user_feature_map.values() for f in feats)
    item_features = set(f for feats in item_feature_map.values() for f in feats) if item_feature_map else []

    dataset.fit(users=user_ids, items=item_ids,
                user_features=user_features,
                item_features=item_features)

    return dataset

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

def train_model(interactions, weights, user_features, item_features):
    from lightfm import LightFM

    model = LightFM(loss="warp", random_state=42)
    model.fit(interactions,
              sample_weight=weights,
              user_features=user_features,
              item_features=item_features,
              epochs=10,
              num_threads=4)
    return model