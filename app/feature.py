import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 한글 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 학습 데이터 구성
data = pd.DataFrame([
    {"skill": "AI", "region": "서울", "target": "경험쌓기", "match": 1.0},
    {"skill": "웹풀스택", "region": "서울", "target": "경험쌓기", "match": 1.0},
    {"skill": "AI", "region": "서울", "target": "상금", "match": 1.0},
    {"skill": "블록체인", "region": "서울", "target": "경험쌓기", "match": 1.0},
    {"skill": "게임서버", "region": "대전", "target": "상금", "match": 0.0},
    {"skill": "프론트엔드", "region": "부산", "target": "1등", "match": 0.0},
    {"skill": "QA", "region": "울산", "target": "상금", "match": 0.0},
    {"skill": "DBA", "region": "제주도", "target": "1등", "match": 0.0},
])

# 특성과 라벨 분리
X = pd.get_dummies(data[["skill", "region", "target"]])
y = data["match"]

# 모델 학습 (Regressor)
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

# 모델 저장
model_path = os.path.join(os.getcwd(), "team_recommender.pkl")
joblib.dump((model, X.columns), model_path)
print("✅ 모델 저장 완료:", model_path)

# 피처 중요도 시각화
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("중요도")
plt.title("📊 팀 추천 모델 피처 중요도 (랜덤 포레스트 회귀)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis='x')

plt.show()
