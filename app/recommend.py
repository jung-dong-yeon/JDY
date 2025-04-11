import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 벡터라이저 로드
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.pkl"))

def get_recommended_teams(user: dict, teams: list):
    results = []

    user_skills = user["skills"]
    user_text = " ".join(user_skills + [user["region"], user["target"]])

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

        # ✅ 보너스 가중치 (최대 0.2)
        skill_bonus = 0.1 if has_skill_match else 0.0
        region_bonus = 0.07 if user["region"] == team["region"] else 0.0
        target_bonus = 0.03 if user["target"] == team["goal"] else 0.0

        final_score = round(sim_score + skill_bonus + region_bonus + target_bonus, 2)

        results.append({
            "team_id": team["team_id"],
            "score": final_score,
            "prediction": 1 if final_score >= 0.5 else 0,
            "badge": "추천" if final_score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
