import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model():
    data = pd.DataFrame([
        {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "AI", "region": "서울", "target": "상금", "match": 1},
        {"skill": "블록체인", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0},
        {"skill": "프론트엔드", "region": "부산", "target": "1등", "match": 0},
        {"skill": "QA", "region": "울산", "target": "상금", "match": 0},
        {"skill": "DBA", "region": "제주도", "target": "1등", "match": 0},
    ])

    # 인코딩
    X_encoded = pd.get_dummies(data[["skill", "region", "target"]])
    y = data["match"]

    # 모델 학습
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X_encoded, y)

    # 저장
    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    joblib.dump((model, X_encoded.columns), model_path)
    print("✅ 모델 저장 완료:", model_path)

if __name__ == "__main__":
    train_model()
