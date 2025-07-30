from collections import defaultdict

def build_user_features(user_brand_df, bookmark_df, brand_df):
    user_feature_map = defaultdict(list)
    bookmark_map = bookmark_df.groupby("user_id")["brand_id"].apply(list).to_dict()
    brand_to_category = dict(zip(brand_df["brand_id"], brand_df["category_id"]))

    for user_id, group in user_brand_df.groupby("user_id"):
        recent = group[group["data_type"] == "RECENT"]["brand_id"].tolist()[:6]
        interest = group[group["data_type"] == "INTEREST"]["brand_id"].tolist()[:3]
        bookmarked = bookmark_map.get(user_id, [])[:5]

        features = (
            [f"recent_{b}" for b in recent] * 2 +
            [f"interest_{b}" for b in interest] * 3 +
            [f"bookmark_{b}" for b in bookmarked] * 2
        )

        category_ids = {brand_to_category.get(b) for b in recent + interest + bookmarked if brand_to_category.get(b)}
        features += [f"cat_{cid}" for cid in category_ids] * 2

        user_feature_map[user_id] = features

    return user_feature_map