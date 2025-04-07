# app/recommend_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("📢 recommend_model.py 실행됨")

def train_model():
    print("✅ 모델 학습 시작")
    data = pd.DataFrame([
        {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "블록체인", "region": "부산", "target": "1등", "match": 0},
        {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0}
    ])
    X = data[["skill", "region", "target"]]
    y = data["match"]
    X_encoded = pd.get_dummies(X)

    model = RandomForestClassifier()
    model.fit(X_encoded, y)

    # 저장 경로 명확하게 고정
    output_path = os.path.abspath("../team_recommender.pkl")
    joblib.dump(model, output_path)
    print("✅ 모델 저장 완료:", output_path)

if __name__ == "__main__":
    train_model()
