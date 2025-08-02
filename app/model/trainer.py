from lightfm import LightFM
from lightfm.data import Dataset

def prepare_dataset(user_df, brand_df, user_feature_map):
    dataset = Dataset()
    dataset.fit(users=user_df["user_id"], items=brand_df["brand_id"])
    all_user_features = set(f for feats in user_feature_map.values() for f in feats)
    dataset.fit_partial(user_features=all_user_features)
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
            dummy_interactions.append((user_id, brand_df["brand_id"].iloc[0], 1.0))

    real = list(zip(interaction_df["user_id"], interaction_df["brand_id"], interaction_df["weight"]))
    return dataset.build_interactions(real + dummy_interactions)

def train_model(interactions, weights, user_features):
    model = LightFM(loss="warp")
    model.fit(interactions, sample_weight=weights, user_features=user_features, epochs=10, num_threads=2)
    return model