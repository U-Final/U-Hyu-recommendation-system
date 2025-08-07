from lightfm.data import Dataset

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

def train_model(interactions, weights, user_features, item_features):
    from lightfm import LightFM

    model = LightFM(loss="warp", random_state=42)
    model.fit(interactions,
              sample_weight=weights,
              user_features=user_features,
              item_features=item_features,
              epochs=100,
              num_threads=4)
    return model