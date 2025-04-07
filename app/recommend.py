# app/recommend.py

import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

def get_recommended_teams(user: dict, teams: list):
    test_rows = []
    team_ids = []

    for team in teams:
        for skill in user["skills"]:
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

    team_scores = {}
    team_preds = {}

    for i, team_id in enumerate(team_ids):
        team_scores.setdefault(team_id, []).append(probs[i])
        team_preds.setdefault(team_id, []).append(preds[i])

    result = [
        {
            "team_id": tid,
            "score": float(round(sum(scores) / len(scores), 2)),
            "prediction": int(round(sum(team_preds[tid]) / len(team_preds[tid])))
        }
        for tid, scores in team_scores.items()
    ]

    return sorted(result, key=lambda x: x["score"], reverse=True)
