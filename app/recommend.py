import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

# 추천 함수
def get_recommended_teams(user: dict, teams: list):
    result = []

    for team in teams:
        # ✅ 필드명 일치: recruitment_skill
        skills = team["recruitment_skill"].split(",")
        skills = [s.strip() for s in skills]

        test_rows = []
        for skill in skills:
            test_rows.append({
                "skill": skill,
                "region": team["region"],
                "target": team["goal"]  # ✅ 'goal'로 받아야 함
            })

        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        probas = model.predict_proba(df_encoded)[:, 1]
        avg_score = float(round(probas.mean(), 2))

        result.append({
            "team_id": team["team_id"],  # ✅ 필드명 일치
            "score": avg_score,
            "prediction": 1 if avg_score > 0.5 else 0,
            "badge": "추천" if avg_score >= 0.6 else ""
        })

    return sorted(result, key=lambda x: x["score"], reverse=True)
