import pandas as pd
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = user.get("skills", [])
    for team in teams:
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        test_rows = []
        team_ids = []

        for user_skill in user_skills:
            for team_skill in team_skills:
                test_rows.append({
                    "skill": team_skill,
                    "region": team["region"],
                    "target": team["goal"]
                })
                team_ids.append(team["team_id"])

        if not test_rows:
            continue

        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        scores = model.predict(df_encoded)
        avg_score = float(round(scores.mean(), 2))

        results.append({
            "team_id": team["team_id"],
            "score": avg_score,
            "prediction": 1 if avg_score >= 0.5 else 0,
            "badge": "추천" if avg_score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
