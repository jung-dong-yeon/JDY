import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 모델 학습 함수
def train_model():
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

    X = pd.get_dummies(data[["skill", "region", "target"]])
    y = data["match"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    joblib.dump((model, X.columns), model_path)
    print("✅ 모델 저장 완료:", model_path)

# 예측 테스트
def predict():
    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    if not os.path.exists(model_path):
        print("❌ 모델 없음")
        return
    model, feature_columns = joblib.load(model_path)

    user_data = {
        "skills": ["AI", "웹풀스택", "블록체인"],
        "region": "서울",
        "target": "경험쌓기"
    }

    teams_data = [
        {"team_id": 1, "recruitment_skill": "AI", "region": "서울", "goal": "경험쌓기"},
        {"team_id": 2, "recruitment_skill": "웹풀스택", "region": "서울", "goal": "경험쌓기"},
        {"team_id": 3, "recruitment_skill": "블록체인", "region": "부산", "goal": "상금"},
        {"team_id": 4, "recruitment_skill": "게임서버", "region": "대전", "goal": "상금"},
    ]

    test_rows = []
    team_ids = []

    for team in teams_data:
        for skill in user_data["skills"]:
            test_rows.append({
                "skill": skill,
                "region": team["region"],
                "target": team["goal"]
            })
            team_ids.append(team["team_id"])

    df = pd.DataFrame(test_rows)
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scores = model.predict(df_encoded)

    score_map = {}
    for i, team_id in enumerate(team_ids):
        score_map.setdefault(team_id, []).append(scores[i])

    result = [
        {
            "team_id": tid,
            "score": round(sum(vals)/len(vals), 2),
            "badge": "추천" if sum(vals)/len(vals) >= 0.6 else ""
        }
        for tid, vals in score_map.items()
    ]

    result = sorted(result, key=lambda x: x["score"], reverse=True)
    print("🔎 추천 결과:")
    for r in result:
        print(r)

if __name__ == "__main__":
    train_model()
    predict()
