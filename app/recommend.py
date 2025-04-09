import pandas as pd
import joblib
import os

# 모델 로드 (모델과 feature columns 함께 저장됨)
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

# 추천 팀 계산 함수
def get_recommended_teams(user: dict, teams: list):
    result = []

    for team in teams:
        # 프론트에서 보내는 필드명 그대로 사용 (skill: 문자열, teamId: 정수)
        skills = team["skill"].split(",")  # 예: "AI, 프론트엔드"
        skills = [s.strip() for s in skills]

        test_rows = []
        for skill in skills:
            test_rows.append({
                "skill": skill,
                "region": team["region"],
                "target": team["target"]
            })

        # 입력 데이터프레임 생성 및 인코딩
        df = pd.DataFrame(test_rows)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # 예측 확률 계산
        probas = model.predict_proba(df_encoded)[:, 1]
        avg_score = float(round(probas.mean(), 2))

        # 추천 결과 저장
        result.append({
            "team_id": team["teamId"],  # ✅ 프론트에서 보내는 teamId 맞춰줌
            "score": avg_score,
            "prediction": 1 if avg_score > 0.5 else 0,
            "badge": "추천" if avg_score >= 0.6 else ""
        })

    # 점수 기준 내림차순 정렬
    return sorted(result, key=lambda x: x["score"], reverse=True)
