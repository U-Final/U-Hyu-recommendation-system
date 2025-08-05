from datetime import datetime

def prepare_statistics_df(recommend_df, brand_df):
    print("📊 통계 데이터 구성 중...")
    statistics_df = recommend_df.merge(
        brand_df[['brand_id', 'brand_name', 'category_id', 'category_name']],
        on='brand_id',
        how='left'
    )

    statistics_df = statistics_df[[
        'user_id', 'brand_id', 'brand_name', 'category_id', 'category_name'
    ]].copy()

    statistics_df['my_map_list_id'] = None
    statistics_df['store_id'] = None
    statistics_df['statistics_type'] = 'RECOMMENDATION'
    statistics_df['created_at'] = datetime.now()
    statistics_df['updated_at'] = datetime.now()

    # 누락된 브랜드 정보 제거
    return statistics_df.dropna(subset=['brand_name', 'category_id', 'category_name'])