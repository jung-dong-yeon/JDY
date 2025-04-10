import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = user.get("skills", [])
    for team in teams:
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        test_rows = []

        for user_skill in user_skills:
            for team_skill in team_skills:
                test_rows.append({
                    "skill": team_skill,
                    "region": team["region"],
                    "target": team["goal"]
                })

        if not test_rows:
            continue

        # DataFrame 생성 및 인코딩
        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # 모델 예측
        scores = model.predict(df_encoded)
        base_score = float(round(scores.mean(), 2))

        # 스킬 겹치는 비율 계산
        match_count = sum(1 for s in user_skills if s.strip() in team_skills)
        match_ratio = match_count / len(user_skills) if user_skills else 0

        # 최종 점수는 예측 평균 * 겹치는 비율
        final_score = round(base_score * match_ratio, 2)

        results.append({
            "team_id": team["team_id"],
            "score": final_score,
            "prediction": 1 if final_score >= 0.5 else 0,
            "badge": "추천" if final_score >= 0.6 else ""
        })

    # 점수 기준 정렬
    return sorted(results, key=lambda x: x["score"], reverse=True)
