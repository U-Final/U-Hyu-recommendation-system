from config.database import get_engine
from data.loader import *
from features.builder import build_user_features
from model.trainer import prepare_dataset, build_interactions, train_model
from model.recommender import generate_recommendations
from saver.db_saver import save_to_db
from saver.file_exporter import save_to_csv

def main():
    print("🚀 추천 시스템 실행 중...")

    # DB 연결
    print("🔌 DB 연결 중...")
    engine = get_engine()
    with engine.connect() as conn:
        print("📥 사용자 데이터 로딩 중...")
        user_df = load_user_data(conn)
        print(f"👤 사용자 수: {len(user_df)}")

        print("📥 브랜드 데이터 로딩 중...")
        brand_df = load_brand_data(conn)
        print(f"🏷️ 브랜드 수: {len(brand_df)}")

        print("📥 온보딩/관심 데이터 로딩 중...")
        user_brand_df = load_user_brand_data(conn)
        print(f"📌 관심 브랜드 수: {len(user_brand_df)}")

        print("📥 인터랙션 데이터 로딩 중...")
        interaction_df = load_interaction_data(conn)
        print(f"🧩 인터랙션 수: {len(interaction_df)}")

        print("📥 즐겨찾기 데이터 로딩 중...")
        bookmark_df = load_bookmark_data(conn)
        print(f"⭐ 즐겨찾기 수: {len(bookmark_df)}")

        print("📥 EXCLUDE 브랜드 로딩 중...")
        exclude_brand_df = load_exclude_brands(conn)
        exclude_brand_ids = set(exclude_brand_df["brand_id"].tolist())
        print(f"🚫 제외 브랜드 수: {len(exclude_brand_ids)}")

    # 피처 생성
    print("🛠️ 사용자 피처 생성 중...")
    user_feature_map = build_user_features(user_brand_df, bookmark_df, brand_df, exclude_brand_ids=exclude_brand_ids)

    print("📦 데이터셋 구성 중...")
    dataset = prepare_dataset(user_df, brand_df, user_feature_map)

    print("🔧 인터랙션 + 가중치 매트릭스 구성 중...")
    interactions, weights = build_interactions(dataset, interaction_df, user_brand_df, brand_df)

    print("🎛️ 사용자 피처 매트릭스 구성 중...")
    user_features = dataset.build_user_features([(uid, feats) for uid, feats in user_feature_map.items()])

    # 모델 학습
    print("🧠 LightFM 모델 학습 중...")
    model = train_model(interactions, weights, user_features)

    # 추천 생성
    print("📊 추천 결과 생성 중...")
    recommend_df = generate_recommendations(user_df, brand_df, model, dataset, user_features, exclude_brand_ids=exclude_brand_ids)
    print(f"🎯 추천 결과 개수: {len(recommend_df)}")

    # DB 저장
    print("💾 추천 결과 DB 저장 중...")
    save_to_db(engine, recommend_df)

    # CSV 저장
    print("📄 추천 결과 CSV 저장 중...")
    save_to_csv(recommend_df)

    print("✅ 추천 완료!")

if __name__ == "__main__":
    main()