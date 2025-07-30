from sqlalchemy import text

def save_to_db(engine, recommend_df):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO recommendation (user_id, brand_id, score, rank, created_at, updated_at)
                VALUES (:user_id, :brand_id, :score, :rank, :created_at, :updated_at)
            """),
            recommend_df[["user_id", "brand_id", "score", "rank", "created_at", "updated_at"]].to_dict("records")
        )