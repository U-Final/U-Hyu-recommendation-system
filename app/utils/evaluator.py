from collections import defaultdict

def evaluate_metrics(recommended, ground_truth, k=5):
    recommended = recommended[:k]
    hits = len(set(recommended) & set(ground_truth))
    precision = hits / k
    recall = hits / len(ground_truth) if ground_truth else 0
    hit = int(hits > 0)
    return {"precision": precision, "recall": recall, "hit": hit}

def evaluate_recommendations(recommend_df, user_brand_df, bookmark_df, interaction_df, brand_df, top_k=5):
    print("\n🧪 추천 결과 평가 중...\n")

    data_type = "INTEREST"
    df = user_brand_df

    # 각 브랜드의 brand_id를 키, category_id를 값으로 하는 딕셔너리를 생성
    brand_category_map = dict(zip(brand_df['brand_id'], brand_df['category_id']))

    # 1. 정답 브랜드 맵
    ground_truth_map = defaultdict(set)
    for row in df.itertuples():
        ground_truth_map[row.user_id].add(row.brand_id)

    # 2. 추천 브랜드 맵
    recommendation_map = defaultdict(list)
    for row in recommend_df.itertuples():
        recommendation_map[row.user_id].append(row.brand_id)

    # 3. 사용자별 Precision/Recall/Hit 계산 및 카테고리 매치율 계산
    scores = defaultdict(list)
    category_match_rates = []

    for user_id in recommend_df['user_id'].unique():
        recommended = recommendation_map.get(user_id, [])[:top_k]
        ground_truth = ground_truth_map.get(user_id, set())

        print(f"user_id: {user_id}")
        print(f"👉 추천 알고리즘이 해당 유저에게 추천한 Top-K 브랜드 리스트 : {recommended}")
        print(f"👉 해당 유저가 실제로 관심 있었다고 판단된 브랜드들: {list(ground_truth)}")

        # 카테고리 매치율 계산
        recommended_categories = set(brand_category_map.get(bid) for bid in recommended if bid in brand_category_map)
        ground_truth_categories = set(brand_category_map.get(bid) for bid in ground_truth if bid in brand_category_map)

        print(f"recommended categories: {list(recommended_categories)}")
        print(f"ground truth categories: {list(ground_truth_categories)}")
        print(f"✅ 추천 브랜드 목록: {[f'{bid} (cat:{brand_category_map.get(bid)})' for bid in recommended]}")
        print(f"✅ 관심 브랜드 목록: {[f'{bid} (cat:{brand_category_map.get(bid)})' for bid in ground_truth]}")

        metrics = evaluate_metrics(recommended, ground_truth, k=top_k)
        for k_metric, v in metrics.items():
            scores[k_metric].append(v)

        if recommended_categories or ground_truth_categories:
            intersection = recommended_categories & ground_truth_categories
            union = recommended_categories | ground_truth_categories
            category_match_rate = len(intersection) / len(union)
        else:
            category_match_rate = 0

        category_match_rates.append(category_match_rate)

    # 4. 평균 계산 및 출력
    avg_metrics = {k_metric: sum(v) / len(v) if v else 0 for k_metric, v in scores.items()}
    avg_category_match_rate = sum(category_match_rates) / len(category_match_rates) if category_match_rates else 0

    print(f"\n📊 INTEREST 추천 평가 지표 : ")
    for k_metric, v in avg_metrics.items():
        print(f"{k_metric}@{top_k}: {v:.4f}")
    print(f"category_match_rate@{top_k}: {avg_category_match_rate:.4f}")
    print()

    # 📌 관심 브랜드 및 카테고리 유사도 기반 예시 사용자 추천 결과 확인
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    print("\n📌 관심 브랜드-추천 결과 카테고리 유사도 기반 예시 사용자 추천 결과 확인 (2~3명):")

    category_vectors = []

    for user_id in recommend_df['user_id'].unique():
        recommended = recommendation_map.get(user_id, [])[:top_k]
        ground_truth = ground_truth_map.get(user_id, set())

        recommended_categories = set(brand_category_map.get(bid) for bid in recommended if bid in brand_category_map)
        ground_truth_categories = set(brand_category_map.get(bid) for bid in ground_truth if bid in brand_category_map)

        all_cats = set(brand_category_map.values())
        vec = []
        for cat in all_cats:
            vec.append(int(cat in recommended_categories))
            vec.append(int(cat in ground_truth_categories))
        category_vectors.append((user_id, vec))

    df_vec = pd.DataFrame([v[1] for v in category_vectors], index=[v[0] for v in category_vectors])
    similarities = cosine_similarity(df_vec)
    avg_sims = similarities.mean(axis=1)
    top_indices = avg_sims.argsort()[::-1][:3]
    top_users = df_vec.index[top_indices]

    for user_id in top_users:
        rec_brands = recommend_df[recommend_df['user_id'] == user_id]['brand_id'].tolist()
        true_brands = user_brand_df[
            (user_brand_df['user_id'] == user_id) &
            (user_brand_df['data_type'].str.upper() == "INTEREST")
        ]['brand_id'].tolist()
        rec_brands_with_cat = [(bid, brand_category_map.get(bid)) for bid in rec_brands]
        print(f"유저 {user_id} 추천 (브랜드, 카테고리) : {rec_brands_with_cat}")
        print(f"유저 {user_id} 관심 브랜드: {true_brands}\n")

    # 📊 사용자별 INTEREST 브랜드 개수 통계
    print("\n📊 사용자별 INTEREST 브랜드 개수 통계:")
    interest_counts = user_brand_df[user_brand_df["data_type"].str.upper() == "INTEREST"].groupby("user_id")['brand_id'].count()
    print(interest_counts.describe())