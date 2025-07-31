import numpy as np
from datetime import datetime, timezone
import pandas as pd

def generate_recommendations(user_df, brand_df, model, dataset, user_features, top_k=5):
    _, _, item_mapping, _ = dataset.mapping()
    index_to_brand_id = {v: k for k, v in item_mapping.items()}
    item_indices = list(index_to_brand_id.keys())

    results = []
    for user_id in user_df["user_id"]:
        user_index = user_df[user_df["user_id"] == user_id].index[0]
        scores = model.predict(
            user_ids=np.repeat(user_index, len(item_indices)),
            item_ids=np.array(item_indices),
            user_features=user_features)
        top_k_indices = np.argsort(-scores)[:top_k]

        for rank, idx in enumerate(top_k_indices, start=1):
            results.append({
                "user_id": user_id,
                "brand_id": index_to_brand_id[item_indices[idx]],
                "score": float(scores[idx]) * 100,
                "rank": rank,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })

    return pd.DataFrame(results)