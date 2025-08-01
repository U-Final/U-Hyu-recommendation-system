from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.model.recommender import generate_recommendation_for_user
from app.config.database import get_engine
from app.data.loader import load_user_data, load_brand_data, load_user_brand_data, load_interaction_data, load_bookmark_data
from app.features.builder import build_user_features
from app.model.trainer import prepare_dataset, build_interactions, train_model
from app.saver.db_saver import save_to_db
import logging
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()

class UserRequest(BaseModel):
    user_id: int

@router.post("/re-recommendation")
def recommend_on_demand(request_body: UserRequest):
    try:
        user_id = request_body.user_id
        logger.info(f"[추천 API] 요청 바디에서 받은 user_id: {user_id}")

        # 1. DB 연결
        try:
            engine = get_engine()
            with engine.connect() as conn:
                user_df = load_user_data(conn, user_ids=[user_id])
                brand_df = load_brand_data(conn)
                user_brand_df = load_user_brand_data(conn, user_ids=[user_id])
                interaction_df = load_interaction_data(conn, user_ids=[user_id])
                bookmark_df = load_bookmark_data(conn, user_ids=[user_id])

                exclude_query = f"""
                    SELECT brand_id
                    FROM recommendation_base_data
                    WHERE user_id = {user_id}
                      AND data_type = 'EXCLUDE'
                """
                exclude_brand_ids = pd.read_sql(exclude_query, conn)["brand_id"].tolist()
                brand_df = brand_df[~brand_df["brand_id"].isin(exclude_brand_ids)]
                interaction_df = interaction_df[~interaction_df["brand_id"].isin(exclude_brand_ids)]
                user_brand_df = user_brand_df[~user_brand_df["brand_id"].isin(exclude_brand_ids)]

        except Exception as e:
            logger.error(f"데이터베이스 연결 또는 데이터 로드 실패: {e}")
            raise HTTPException(status_code=503, detail="데이터베이스 연결 실패") from e

        # 2. 사용자 피쳐 구성
        user_feature_map = build_user_features(user_brand_df, bookmark_df, brand_df)

        # 3. dataset 구성
        dataset = prepare_dataset(user_df, brand_df, user_feature_map)
        interactions, weights = build_interactions(dataset, interaction_df, user_brand_df, brand_df)
        user_features = dataset.build_user_features([(uid, feats) for uid, feats in user_feature_map.items()])

        # 4. 모델 학습
        model = train_model(interactions, weights, user_features)

        # 5. 추천 생성
        recommend_df = generate_recommendation_for_user(
            user_id, user_df, brand_df, model, dataset, user_features
        )

        if recommend_df.empty:
            raise HTTPException(status_code=404, detail="추천할 브랜드가 없습니다.")

        # 6. DB 저장
        save_to_db(engine, recommend_df)

        # 7. 응답 반환
        return {
            "user_id": user_id,
            "recommendations": recommend_df[["brand_id", "score", "rank"]].to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("추천 생성 중 오류 발생", exc_info=True)
        raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.") from e