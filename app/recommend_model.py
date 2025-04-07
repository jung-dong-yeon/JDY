import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    data = pd.DataFrame([
        {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "블록체인", "region": "부산", "target": "1등", "match": 0},
        {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기", "match": 1},
        {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0},
    ])

    X = data[["skill", "region", "target"]]
    y = data["match"]
    X_encoded = pd.get_dummies(X)

    model = RandomForestClassifier()
    model.fit(X_encoded, y)

    joblib.dump(model, "team_recommender.pkl")
    print("✅ 모델 저장 완료: team_recommender.pkl")

    
if __name__ == "__main__":
    train_model()
