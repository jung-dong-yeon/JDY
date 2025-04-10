import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = user.get("skills", [])
    user_region = user.get("region", "")
    user_target = user.get("target", "")

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

        # 예측용 데이터 준비
        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # 랜덤포레스트 예측 점수
        base_score = float(round(model.predict(df_encoded).mean(), 2))

        # 유사도 계산
        skill_match_ratio = len(set(user_skills) & set(team_skills)) / max(len(user_skills), 1)
        region_match = 1.0 if user_region == team["region"] else 0.0
        target_match = 1.0 if user_target == team["goal"] else 0.0

        # 🔥 조정된 가중치 기반 최종 점수
        final_score = round(
            (0.45 * base_score) +
            (0.30 * skill_match_ratio) +
            (0.15 * region_match) +
            (0.10 * target_match),
            2
        )

        final_score = round(
    (0.2 * base_score) + (0.5 * skill_match_ratio) + (0.2 * region_match) + (0.1 * target_match),
    2
)


    return sorted(results, key=lambda x: x["score"], reverse=True)
