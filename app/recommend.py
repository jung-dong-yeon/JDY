import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 벡터라이저 로드
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.pkl"))

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = user["skills"]
    user_region = user["region"]
    user_target = user["target"]
    user_text = " ".join(user_skills + [user_region, user_target])

    team_texts = [
        " ".join(
            [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
            + [team["region"], team["goal"]]
        )
        for team in teams
    ]

    all_texts = [user_text] + team_texts
    vectors = vectorizer.transform(all_texts)

    user_vector = vectors[0]
    team_vectors = vectors[1:]

    similarities = cosine_similarity(user_vector, team_vectors)[0]

    for team, sim_score in zip(teams, similarities):
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        has_skill_match = bool(set(user_skills) & set(team_skills))
        region_match = user_region == team["region"]
        target_match = user_target == team["goal"]

        # ✅ 가중치 설정
        skill_bonus = 0.25 if has_skill_match else 0.0
        region_bonus = 0.25 if region_match else 0.0
        target_bonus = 0.10 if target_match else 0.0

        # ✅ 코사인 유사도 최대치 제한
        sim_score = min(sim_score, 0.40)

        final_score = round(sim_score + skill_bonus + region_bonus + target_bonus, 2)

        results.append({
            "team_id": team["team_id"],
            "score": final_score,
            "prediction": 1 if final_score >= 0.5 else 0,
            "badge": "추천" if final_score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
