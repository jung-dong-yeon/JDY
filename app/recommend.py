import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = [s.strip() for s in user.get("skills", []) if s.strip()]
    if not user_skills:
        return []  # 유저 스킬이 없으면 추천 불가

    for team in teams:
        team_skills = [s.strip() for s in team.get("recruitment_skill", "").split(",") if s.strip()]
        if not team_skills:
            continue

        test_rows = []
        for user_skill in user_skills:
            for team_skill in team_skills:
                test_rows.append({
                    "skill": team_skill,
                    "region": team.get("region", ""),
                    "target": team.get("goal", "")
                })

        if not test_rows:
            continue

        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        scores = model.predict(df_encoded)
        base_score = float(round(scores.mean(), 4))  # ⬅ 소수점 4자리까지 계산하고 최종만 round(2)

        match_count = sum(1 for s in user_skills if s in team_skills)
        match_ratio = match_count / len(user_skills)

        final_score = round(base_score * match_ratio, 2)

        results.append({
            "team_id": team.get("team_id"),
            "score": final_score,
            "prediction": 1 if final_score >= 0.5 else 0,
            "badge": "추천" if final_score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
