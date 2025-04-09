# train.py (모델 학습 파일)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 다양한 조합의 학습 데이터 생성
data = []
skills = ["AI", "웹풀스택", "게임서버", "프론트엔드", "백엔드", "블록체인", "QA"]
regions = ["서울", "대전", "부산", "울산", "제주도"]
targets = ["경험쌓기", "상금", "1등"]

for skill in skills:
    for region in regions:
        for target in targets:
            match = 1 if skill in ["AI", "웹풀스택"] and region == "서울" and target in ["상금", "1등"] else 0
            data.append({
                "skill": skill,
                "region": region,
                "target": target,
                "match": match
            })

df = pd.DataFrame(data)
X = pd.get_dummies(df[["skill", "region", "target"]])
y = df["match"]

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X, y)

# 모델과 feature 컬럼 저장
model_path = os.path.join(os.getcwd(), "team_recommender.pkl")
joblib.dump((model, X.columns), model_path)

print("✅ 모델 학습 완료")
