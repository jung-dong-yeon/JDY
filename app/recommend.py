import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 벡터라이저 로드 (Render 배포용 - 상대경로 처리)
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    # 유저 텍스트: skills + region + target
    user_text = " ".join(user["skills"] + [user["region"], user["target"]])

    # 각 팀의 텍스트: recruitment_skill + region + goal
    team_texts = [
        " ".join(
            [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
            + [team["region"], team["goal"]]
        )
        for team in teams
    ]

    # 벡터화
    all_texts = [user_text] + team_texts
    vectors = vectorizer.transform(all_texts)

    user_vector = vectors[0]
    team_vectors = vectors[1:]

    # 유사도 계산
    similarities = cosine_similarity(user_vector, team_vectors)[0]

    for team, sim_score in zip(teams, similarities):
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        skill_match_ratio = len(set(user["skills"]) & set(team_skills)) / max(len(team_skills), 1)

        # ✅ 유사도 85% + 스킬 매칭 15%
        score = round((0.87 * sim_score) + (0.13 * skill_match_ratio), 2)

        results.append({
            "team_id": team["team_id"],
            "score": score,
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    # 높은 점수 기준 내림차순 정렬
    return sorted(results, key=lambda x: x["score"], reverse=True)
