# app/recommend.py
import pandas as pd
import joblib
import os

# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

# 추천 팀 계산 함수
def get_recommended_teams(user: dict, teams: list):
    test_rows = []  # 팀과 유저 데이터를 비교할 행을 추가
    team_ids = []   # 팀 ID 저장

    # 각 팀과 유저의 기술, 지역, 목표 비교
    for team in teams:
        for skill in user["skills"]:
            test_rows.append({
                "skill": skill,
                "region": team["region"],
                "target": team["goal"]
            })
            team_ids.append(team["team_id"])

    # 데이터프레임 생성
    X_test = pd.DataFrame(test_rows)
    X_test_encoded = pd.get_dummies(X_test)  # 원핫 인코딩
    X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)

    # 예측 확률 및 예측값
    probs = model.predict_proba(X_test_encoded)[:, 1]  # 확률
    preds = model.predict(X_test_encoded)  # 이진 분류 결과

    team_scores = {}
    team_preds = {}

    # 팀별 점수 및 예측 결과 계산
    for i, team_id in enumerate(team_ids):
        team_scores.setdefault(team_id, []).append(probs[i])
        team_preds.setdefault(team_id, []).append(preds[i])

    # 결과 형성
    result = [
        {
            "team_id": tid,
            "score": float(round(sum(scores) / len(scores), 2)),
            "prediction": int(round(sum(team_preds[tid]) / len(team_preds[tid]))),
            "badge": "추천" if (sum(scores) / len(scores)) >= 0.8 else ""  # ✅ 추천 뱃지 조건
        }
        for tid, scores in team_scores.items()
    ]

    # 점수 기준으로 내림차순 정렬
    return sorted(result, key=lambda x: x["score"], reverse=True)
