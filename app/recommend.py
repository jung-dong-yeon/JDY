import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 벡터라이저 로드
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    # 유저 텍스트: 스킬 + 지역 + 목표
    user_text = " ".join(user["skills"] + [user["region"], user["target"]])

    # 팀 텍스트들
    team_texts = [
        " ".join(
            [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
            + [team["region"], team["goal"]]
        ) for team in teams
    ]

    # 전체 벡터화
    all_texts = [user_text] + team_texts
    vectors = vectorizer.transform(all_texts)

    user_vector = vectors[0]
    team_vectors = vectors[1:]
    similarities = cosine_similarity(user_vector, team_vectors)[0]

    for team, sim_score in zip(teams, similarities):
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        skill_match_ratio = len(set(user["skills"]) & set(team_skills)) / max(len(team_skills), 1)
        region_match = 1.0 if user["region"] == team["region"] else 0.0
       
        # ✅ 점수 계산
        score = round(
            (0.8 * sim_score) +
            (0.15 * skill_match_ratio) +
            (0.05 * region_match),
            2
        )
        score = min(score, 0.9)  # 최대 점수 제한

        results.append({
            "team_id": team["team_id"],
            "score": score,
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
