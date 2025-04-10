import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer

def train_vectorizer():
    data = pd.DataFrame([
        {"skill": "AI", "region": "서울", "target": "경험쌓기"},
        {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기"},
        {"skill": "AI", "region": "서울", "target": "상금"},
        {"skill": "블록체인", "region": "서울", "target": "경험쌓기"},
        {"skill": "게임서버", "region": "대전", "target": "상금"},
        {"skill": "프론트엔드", "region": "부산", "target": "1등"},
        {"skill": "QA", "region": "울산", "target": "상금"},
        {"skill": "DBA", "region": "제주도", "target": "1등"},
    ])
    texts = data.apply(lambda row: f"{row['skill']}, {row['region']}, {row['target']}", axis=1)
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✅ 벡터라이저 저장 완료")

if __name__ == "__main__":
    train_vectorizer()
