from sqlalchemy import text
import logging
logger = logging.getLogger(__name__)

def save_to_db(engine, recommend_df):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO recommendation (user_id, brand_id, score, rank, created_at, updated_at)
                VALUES (:user_id, :brand_id, :score, :rank, :created_at, :updated_at)
            """),
            recommend_df[["user_id", "brand_id", "score", "rank", "created_at", "updated_at"]].to_dict("records")
        )

def save_statistics(engine, statistics_df):
    if statistics_df.empty:
        logger.warning("⚠️ 저장할 통계 데이터가 없습니다.")
        return

    # statistics_records = []
    # for row in statistics_df.itertuples(index=False):
    #     if not all([row.brand_name, row.category_id, row.category_name]):
    #         continue  # 누락된 정보가 있다면 저장하지 않음
    #
    #     statistics_records.append({
    #         "user_id": row.user_id,
    #         "my_map_list_id": None,
    #         "store_id": None,
    #         "brand_id": row.brand_id,
    #         "brand_name": row.brand_name,
    #         "category_id": row.category_id,
    #         "category_name": row.category_name,
    #         "statistics_type": "RECOMMENDATION",
    #         "created_at": row.created_at,
    #         "updated_at": row.updated_at,
    #     })

    statistics_records = statistics_df.to_dict('records')

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO statistics (
                    user_id, my_map_list_id, store_id,
                    brand_id, brand_name, category_id, category_name,
                    statistics_type, created_at, updated_at
                )
                VALUES (
                    :user_id, :my_map_list_id, :store_id,
                    :brand_id, :brand_name, :category_id, :category_name,
                    :statistics_type, :created_at, :updated_at
                )
            """),
            statistics_records
        )

    logger.info(f"📊 통계 {len(statistics_records)}건 저장 완료")