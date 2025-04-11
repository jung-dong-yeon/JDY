import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 벡터라이저 로드
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    # ✅ 유저 텍스트 생성
    user_skills = user.get("skills", [])
    user_region = user.get("region", "")
    user_target = user.get("target", "")
    user_text = " ".join(user_skills + [user_region, user_target])

    # ✅ 팀 텍스트 리스트 생성
    team_texts = []
    for team in teams:
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        team_text = " ".join(team_skills + [team["region"], team["goal"]])
        team_texts.append(team_text)

    # ✅ 벡터화
    all_texts = [user_text] + team_texts
    vectors = vectorizer.transform(all_texts)
    user_vector = vectors[0]
    team_vectors = vectors[1:]

    # ✅ 유사도 계산
    similarities = cosine_similarity(user_vector, team_vectors)[0]

    # ✅ 결과 구성
    for team, score in zip(teams, similarities):
        results.append({
            "team_id": team["team_id"],
            "score": round(float(score), 2),
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    # ✅ 유사도 기준 내림차순 정렬
    return sorted(results, key=lambda x: x["score"], reverse=True)
