from collections import defaultdict

def evaluate_metrics(recommended, ground_truth, k=5):
    recommended = recommended[:k]
    hits = len(set(recommended) & set(ground_truth))
    precision = hits / k
    recall = hits / len(ground_truth) if ground_truth else 0
    hit = int(hits > 0)
    return {"precision": precision, "recall": recall, "hit": hit}

def evaluate_recommendations(recommend_df, user_brand_df, bookmark_df, interaction_df, top_k=5):
    print("🧪 추천 결과 평가 중...")

    data_types = {
        "INTEREST": user_brand_df,
        "BOOKMARK": bookmark_df,
        "HISTORY": interaction_df
    }

    for data_type, df in data_types.items():
        # 1. 정답 브랜드 맵
        ground_truth_map = defaultdict(set)
        for row in df.itertuples():
            if getattr(row, 'data_type', data_type) == data_type:
                ground_truth_map[row.user_id].add(row.brand_id)

        # 2. 추천 브랜드 맵
        recommendation_map = defaultdict(list)
        for row in recommend_df.itertuples():
            recommendation_map[row.user_id].append(row.brand_id)

        # 3. 사용자별 Precision/Recall/Hit 계산
        scores = defaultdict(list)

        for user_id in recommend_df['user_id'].unique():
            recommended = recommendation_map.get(user_id, [])[:top_k]
            ground_truth = ground_truth_map.get(user_id, set())

            metrics = evaluate_metrics(recommended, ground_truth, k=top_k)
            for k_metric, v in metrics.items():
                scores[k_metric].append(v)

        # 4. 평균 계산 및 출력
        avg_metrics = {k_metric: sum(v) / len(v) if v else 0 for k_metric, v in scores.items()}
        print(f"📊 {data_type} 추천 평가 지표 : ")
        for k_metric, v in avg_metrics.items():
            print(f"{k_metric}@{top_k}: {v:.4f}")
        print()