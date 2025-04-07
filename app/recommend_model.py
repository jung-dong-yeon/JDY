import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 예시 학습 데이터
data = pd.DataFrame([
    {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1},
    {"skill": "블록체인", "region": "부산", "target": "1등", "match": 0},
    {"skill": "AI", "region": "서울", "target": "1등", "match": 1},
    {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0},
])

# 특성과 라벨 분리
X = data[["skill", "region", "target"]]
y = data["match"]

# 범주형 처리
X_encoded = pd.get_dummies(X)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_encoded, y)

# 저장
joblib.dump(model, "team_recommender.pkl")
