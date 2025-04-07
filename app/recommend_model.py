import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    # 학습 데이터 준비
    data = pd.DataFrame([
        {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "블록체인", "region": "부산", "target": "1등", "match": 0},
        {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0},
        {"skill": "AI", "region": "서울", "target": "상금", "match": 1},
        {"skill": "블록체인", "region": "서울", "target": "경험쌓기", "match": 1}
    ])

    # 특성과 라벨 분리
    X = data[["skill", "region", "target"]]
    y = data["match"]

    # 범주형 변수 처리 (one-hot encoding)
    X_encoded = pd.get_dummies(X)

    # 모델 학습
    model = RandomForestClassifier()
    model.fit(X_encoded, y)

    # 모델 저장
    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    joblib.dump(model, model_path)
    print("✅ 모델 저장 완료:", model_path)

if __name__ == "__main__":
    train_model()
