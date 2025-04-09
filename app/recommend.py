import pandas as pd
import joblib
import os
import json

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

# 추천 함수
def get_recommended_teams(user: dict, teams: list):
    result = []

    # 🛠️ userSkill 파싱 (문자열로 오면 JSON으로 변환)
    user_skills = user.get("userSkill", [])
    if isinstance(user_skills, str):
        try:
            user_skills = json.loads(user_skills)
        except:
            user_skills = [s.strip() for s in user_skills.split(",")]

    for team in teams:
        # ✅ 팀에서 모집하는 기술 분리
        skills = team["recruitment_skill"].split(",")
        skills = [s.strip() for s in skills]

        test_rows = []
        for skill in skills:
            test_rows.append({
                "skill": skill,
                "region": team["region"],
                "target": team["goal"]
            })

        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)

        # ✅ 모델이 학습한 feature 컬럼 기준으로 정렬
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # ✅ 각 row에 대한 추천 확률 예측 후 평균
        probas = model.predict_proba(df_encoded)[:, 1]
        avg_score = float(round(probas.mean(), 2))

        result.append({
            "team_id": team["team_id"],
            "score": avg_score,
            "prediction": 1 if avg_score > 0.5 else 0,
            "badge": "추천" if avg_score >= 0.6 else ""
        })

    # ✅ 높은 점수 순 정렬
    return sorted(result, key=lambda x: x["score"], reverse=True)
