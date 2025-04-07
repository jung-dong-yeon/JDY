import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

    X = data[["skill", "region", "target"]]
    y = data["match"]
    X_encoded = pd.get_dummies(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_encoded, y)

    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    joblib.dump((model, X_encoded.columns), model_path)
    print("✅ 모델 저장 완료:", model_path)


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
    if os.path.exists(model_path):
        model, columns = joblib.load(model_path)
        print(f"✅ 모델 로드 완료: {model_path}")
        return model, columns
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None, None


def predict():
    model, feature_columns = load_model()
    if not model:
        return

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

    X_test = pd.DataFrame(test_rows)
    X_test_encoded = pd.get_dummies(X_test)
    X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)

    probs = model.predict_proba(X_test_encoded)[:, 1]
    preds = model.predict(X_test_encoded)

    # 팀별 점수 및 예측값 평균 계산
    team_scores = {}
    team_preds = {}

    for i, team_id in enumerate(team_ids):
        team_scores.setdefault(team_id, []).append(probs[i])
        team_preds.setdefault(team_id, []).append(preds[i])

    result = [
        {
            "team_id": tid,
            "score": round(sum(scores) / len(scores), 2),
            "prediction": int(round(sum(team_preds[tid]) / len(team_preds[tid])))
        }
        for tid, scores in team_scores.items()
    ]

    result = sorted(result, key=lambda x: x["score"], reverse=True)

    print("🔎 추천 결과:")
    for r in result:
        print(r)


if __name__ == "__main__":
    train_model()
    predict()
