import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 벡터라이저 로드
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.pkl"))

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_text = " ".join(user["skills"] + [user["region"], user["target"]])
    team_texts = [
        " ".join(
            [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
            + [team["region"], team["goal"]]
        ) for team in teams
    ]

    # 전체 텍스트 벡터화
    all_texts = [user_text] + team_texts
    vectors = vectorizer.transform(all_texts)

    user_vector = vectors[0]
    team_vectors = vectors[1:]

    # 코사인 유사도 계산
    similarities = cosine_similarity(user_vector, team_vectors)[0]

    # 결과 정리
    for team, score in zip(teams, similarities):
        results.append({
            "team_id": team["team_id"],
            "score": round(float(score), 2),
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
