from datetime import datetime
from app.saver.db_saver import save_statistics
from app.config.database import get_engine
from app.data.loader import *
from app.features.builder import build_user_features, build_item_features, build_interactions
from app.model.trainer import prepare_dataset, train_model
from app.model.recommender import generate_recommendations
from app.saver.db_saver import save_to_db
from app.utils.statistics import prepare_statistics_df
from app.saver.file_exporter import save_to_csv
from app.utils.evaluator import evaluate_recommendations

def main():
    print("🚀 추천 시스템 실행 중...")

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

    print("🛠️ 아이템 피처 생성 중...")
    item_feature_map = build_item_features(brand_df)
    dataset = prepare_dataset(user_df, brand_df, user_feature_map, item_feature_map)
    item_features = dataset.build_item_features([(iid, feats) for iid, feats in item_feature_map.items()])

    print("🔧 인터랙션 + 가중치 매트릭스 구성 중...")
    interactions, weights = build_interactions(dataset, interaction_df, user_brand_df, brand_df)

    print("🎛️ 사용자 피처 매트릭스 구성 중...")
    user_features = dataset.build_user_features([(uid, feats) for uid, feats in user_feature_map.items()])

    # 모델 학습
    print("🧠 LightFM 모델 학습 중...")
    # model = train_model(interactions, weights, user_features)
    model = train_model(interactions, weights, user_features, item_features)

    # 추천 생성
    print("📊 추천 결과 생성 중...")
    recommend_df = generate_recommendations(
        user_df, brand_df, model, dataset,
        user_features, item_features,
        exclude_brand_ids=exclude_brand_ids
    )
    print(f"🎯 추천 결과 개수: {len(recommend_df)}")

    # 5. 추천 평가
    # evaluate_recommendations(recommend_df, user_brand_df, bookmark_df, interaction_df, brand_df)
    evaluate_recommendations(recommend_df, user_brand_df, brand_df)

    # DB 저장
    print("💾 추천 결과 DB 저장 중...")
    save_to_db(engine, recommend_df)

    # CSV 저장
    # print("📄 추천 결과 CSV 저장 중...")
    # save_to_csv(recommend_df)

    # 📊 통계용 데이터 구성
    statistics_df = prepare_statistics_df(recommend_df, brand_df)

    # DB에 통계 저장
    try:
        print("📥 통계 데이터 저장 중...")
        save_statistics(engine, statistics_df)
    except Exception as e:
        print(f"❌ 통계 저장 중 오류 발생: {e}")

    print("✅ 추천 완료!")

if __name__ == "__main__":
    main()