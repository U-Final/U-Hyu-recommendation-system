from config.database import get_engine
from data.loader import *
from features.builder import build_user_features
from model.trainer import prepare_dataset, build_interactions, train_model
from model.recommender import generate_recommendations
from saver.db_saver import save_to_db
from saver.file_exporter import save_to_csv

def main():
    print("🚀 추천 시스템 실행 중...")
    engine = get_engine()
    with engine.connect() as conn:
        user_df = load_user_data(conn)
        brand_df = load_brand_data(conn)
        user_brand_df = load_user_brand_data(conn)
        interaction_df = load_interaction_data(conn)
        bookmark_df = load_bookmark_data(conn)

    user_feature_map = build_user_features(user_brand_df, bookmark_df, brand_df)
    dataset = prepare_dataset(user_df, brand_df, user_feature_map)
    interactions, weights = build_interactions(dataset, interaction_df, user_brand_df, brand_df)
    user_features = dataset.build_user_features([(uid, feats) for uid, feats in user_feature_map.items()])
    model = train_model(interactions, weights, user_features)
    recommend_df = generate_recommendations(user_df, brand_df, model, dataset, user_features)
    save_to_db(engine, recommend_df)
    save_to_csv(recommend_df)
    print("✅ 추천 완료")

if __name__ == "__main__":
    main()