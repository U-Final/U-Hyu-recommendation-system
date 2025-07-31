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

import numpy as np
import pandas as pd
from datetime import datetime, timezone

def generate_recommendation_for_user(user_id, user_df, brand_df, model, dataset, user_features, top_k=5):
    """
    Generate recommendations for a single user using a trained LightFM model.
    """
    _, _, item_mapping, _ = dataset.mapping()
    index_to_brand_id = {v: k for k, v in item_mapping.items()}
    item_indices = list(index_to_brand_id.keys())

    # user_id가 user_df에 없으면 빈 DataFrame 반환
    if user_id not in user_df["user_id"].values:
        return pd.DataFrame([])

    # user_index = user_df에서 해당 user_id의 인덱스를 가져오기
    user_index = user_df[user_df["user_id"] == user_id].index[0]

    # 예측 score 계산
    scores = model.predict(
        user_ids=np.repeat(user_index, len(item_indices)),
        item_ids=np.array(item_indices),
        user_features=user_features
    )

    # 상위 top_k 인덱스 추출
    top_k_indices = np.argsort(-scores)[:top_k]

    # 추천 결과 생성
    results = []
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