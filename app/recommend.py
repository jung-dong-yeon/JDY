import pandas as pd
import joblib
import os

# 🔹 모델 로드: 저장된 모델(.pkl) 파일과 feature_columns(학습 시 사용한 컬럼 정보)를 불러옴
model_path = os.path.join(os.path.dirname(__file__), "team_recommender.pkl")
model, feature_columns = joblib.load(model_path)

# 🔹 팀 추천 함수: 사용자 정보와 팀 리스트를 받아서 가장 적합한 팀을 추천함
def get_recommended_teams(user: dict, teams: list):
    test_rows = []  # 테스트할 입력 데이터를 저장할 리스트  
    team_ids = []   # 각 입력 행에 해당하는 팀 ID 저장

    # 🔸 사용자 스킬과 팀 정보를 조합하여 테스트 데이터 구성
    for team in teams:
        for skill in user["skills"]:
            test_rows.append({
                "skill": skill,             # 사용자 기술 1개
                "region": team["region"],   # 팀 지역
                "target": team["goal"]      # 팀 목표
            })
            team_ids.append(team["team_id"])  # 해당 행이 어떤 팀인지 추적

    # 🔸 pandas DataFrame으로 변환 후 One-hot 인코딩
    X_test = pd.DataFrame(test_rows)
    X_test_encoded = pd.get_dummies(X_test)

    # 🔸 학습된 모델이 사용한 feature 컬럼 순서에 맞춰 정렬 (없는 컬럼은 0으로 채움)
    X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)

    # 🔸 예측 확률 및 결과 계산
    probs = model.predict_proba(X_test_encoded)[:, 1]  # 각 행의 양성 클래스 확률
    preds = model.predict(X_test_encoded)              # 각 행의 이진 예측값 (0 또는 1)

    # 🔸 팀별로 결과를 집계하기 위한 딕셔너리
    team_scores = {}
    team_preds = {}

    # 각 행에 대한 예측 결과를 팀별로 그룹핑
    for i, team_id in enumerate(team_ids):
        team_scores.setdefault(team_id, []).append(probs[i])  # 확률 평균용
        team_preds.setdefault(team_id, []).append(preds[i])   # 예측 평균용

    # 🔸 팀별로 평균 점수와 예측값을 계산
    result = [
        {
            "team_id": tid,
            "score": float(round(sum(scores) / len(scores), 2)),             # 예측 확률 평균
            "prediction": int(round(sum(team_preds[tid]) / len(team_preds[tid])))  # 예측값 다수결
        }
        for tid, scores in team_scores.items()
    ]

    # 🔸 점수가 높은 순으로 정렬하여 추천 결과 반환
    return sorted(result, key=lambda x: x["score"], reverse=True)
