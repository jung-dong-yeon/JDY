import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# 샘플 학습 데이터
data = pd.DataFrame([   
    {"skill": "프론트엔드", "region": "서울", "target": "경험쌓기"},
    {"skill": "게임 서버", "region": "광주", "target": "상금"},
    {"skill": "그래픽스", "region": "서울", "target": "수상"},
    {"skill": "머신러닝", "region": "부산", "target": "포트폴리오"},
])

# 텍스트 조합
texts = data.apply(lambda row: " ".join([row["skill"], row["region"], row["target"]]), axis=1)

# TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

# 저장
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)
print("✅ vectorizer 저장 완료:", vectorizer_path)
