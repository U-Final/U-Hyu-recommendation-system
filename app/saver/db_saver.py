from sqlalchemy import text

def save_to_db(engine, recommend_df, brand_df, category_df):

    # 1. brand 정보 merge
    df = recommend_df.merge(brand_df, left_on="brand_id", right_on="brand_id", how="left")

    # 2. category 정보 merge
    df = df.merge(category_df, left_on="category_id", right_on="category_id", how="left")

    # 3. recommendation 저장
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO recommendation (user_id, brand_id, score, rank, created_at, updated_at)
                VALUES (:user_id, :brand_id, :score, :rank, :created_at, :updated_at)
            """),
            recommend_df[["user_id", "brand_id", "score", "rank", "created_at", "updated_at"]].to_dict("records")
        )

        # 4. statistics 저장
        conn.execute(
            text("""
                INSERT INTO statistics (
                    user_id, brand_id, brand_name,
                    category_id, category_name,
                    statistics_type, created_at, updated_at
                )
                VALUES (
                    :user_id, :brand_id, :brand_name,
                    :category_id, :category_name,
                    :statistics_type, :created_at, :updated_at
                )
            """),
            df.assign(
                statistics_type="RECOMMENDATION"
            )[[
                "user_id", "brand_id", "brand_name",
                "category_id", "category_name",
                "statistics_type", "created_at", "updated_at"
            ]].to_dict("records")
        )