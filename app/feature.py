import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정

model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, columns = joblib.load(model_path)

importances = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(columns, importances)
plt.xlabel("중요도")
plt.title("📊 팀 추천 모델 피처 중요도")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()
