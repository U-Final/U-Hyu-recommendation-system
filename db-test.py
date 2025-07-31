import psycopg2

conn = psycopg2.connect(
    dbname="testdb",
    user="testuser",
    password="testpass",
    host="localhost",
    port=5432
)

cur = conn.cursor()
cur.execute("SELECT * from brands")
print(cur.fetchone())

cur.close()
conn.close()

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

# 1. .env 파일 로드
load_dotenv()

# 2. 환경 변수에서 DB 접속 정보 읽기
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# 3. SQLAlchemy DB URL 구성
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"✅ Connecting to: {DB_URL}")

# 4. DB 연결 테스트
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW()"))
        print("🎉 DB 연결 성공:", result.fetchone()[0])
except Exception as e:
    print("❌ DB 연결 실패:", e)