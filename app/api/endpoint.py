from fastapi import APIRouter, HTTPException
from app.model.recommender import generate_recommendation_for_user
from app.config.database import get_engine
from app.data.loader import load_user_data, load_brand_data, load_user_brand_data, load_interaction_data, load_bookmark_data
from app.features.builder import build_user_features
from app.model.trainer import prepare_dataset, build_interactions, train_model
from app.saver.db_saver import save_to_db

router = APIRouter()

@router.get("/recommend-on-demand/{user_id}")
def recommend_on_demand(user_id: int):
    try:
        # 1. DB 연결
        engine = get_engine()
        with engine.connect() as conn:
            user_df = load_user_data(conn, user_ids=[user_id])
            brand_df = load_brand_data(conn)
            user_brand_df = load_user_brand_data(conn, user_ids=[user_id])
            interaction_df = load_interaction_data(conn, user_ids=[user_id])
            bookmark_df = load_bookmark_data(conn, user_ids=[user_id])

        # 2. 사용자 피쳐 구성
        user_feature_map = build_user_features(user_brand_df, bookmark_df, brand_df)

        # 3. dataset 구성
        dataset = prepare_dataset(user_df, brand_df, user_feature_map)
        interactions, weights = build_interactions(dataset, interaction_df, user_brand_df, brand_df)
        user_features = dataset.build_user_features([(uid, feats) for uid, feats in user_feature_map.items()])

        # 4. 모델 학습
        model = train_model(interactions, weights, user_features)

        # 3. 추천 생성
        recommend_df = generate_recommendation_for_user(
            user_id, user_df, brand_df, model, dataset, user_features
        )

        if recommend_df.empty:
            raise HTTPException(status_code=404, detail="추천할 브랜드가 없습니다.")

        # 4. DB 저장
        save_to_db(engine, recommend_df)

        # 5. 응답 반환
        return {
            "user_id": user_id,
            "recommendations": recommend_df[["brand_id", "score", "rank"]].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))