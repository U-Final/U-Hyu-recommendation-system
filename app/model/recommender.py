import numpy as np
from datetime import datetime, timezone
import pandas as pd

def _predict_user_scores(user_index, model, item_indices, user_features, item_features):
    scores = model.predict(
        user_ids=np.repeat(user_index, len(item_indices)),
        item_ids=np.array(item_indices),
        user_features=user_features,
        item_features=item_features
    )
    return scores

def _build_recommendation_result(user_id, scores, item_indices, index_to_brand_id, top_k):
    top_k_indices = np.argsort(-scores)[:top_k]
    now = datetime.now(timezone.utc)
    return [
        {
            "user_id": user_id,
            "brand_id": index_to_brand_id[item_indices[idx]],
            "score": float(scores[idx]) * 100,
            "rank": rank,
            "created_at": now,
            "updated_at": now
        }
        for rank, idx in enumerate(top_k_indices, start=1)
    ]

def generate_recommendations(user_df, brand_df, model, dataset, user_features, item_features, top_k=5, exclude_brand_ids=None):
    _, _, item_mapping, _ = dataset.mapping()
    index_to_brand_id = {v: k for k, v in item_mapping.items()}

    if exclude_brand_ids:
        item_indices = [idx for idx in index_to_brand_id if index_to_brand_id[idx] not in exclude_brand_ids]
    else:
        item_indices = list(index_to_brand_id.keys())

    results = []
    for user_id in user_df["user_id"]:
        user_index = user_df[user_df["user_id"] == user_id].index[0]
        scores = _predict_user_scores(user_index, model, item_indices, user_features, item_features)
        results.extend(_build_recommendation_result(user_id, scores, item_indices, index_to_brand_id, top_k))

    return pd.DataFrame(results)

def generate_recommendation_for_user(user_id, user_df, brand_df, model, dataset, user_features, item_features, top_k=5, exclude_brand_ids=None):
    _, _, item_mapping, _ = dataset.mapping()
    index_to_brand_id = {v: k for k, v in item_mapping.items()}

    if exclude_brand_ids:
        item_indices = [idx for idx in index_to_brand_id if index_to_brand_id[idx] not in exclude_brand_ids]
    else:
        item_indices = list(index_to_brand_id.keys())

    if user_id not in user_df["user_id"].values:
        return pd.DataFrame([])

    user_index = user_df[user_df["user_id"] == user_id].index[0]
    scores = _predict_user_scores(user_index, model, item_indices, user_features, item_features)
    results = _build_recommendation_result(user_id, scores, item_indices, index_to_brand_id, top_k)

    return pd.DataFrame(results)