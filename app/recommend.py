import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 벡터라이저 로드
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    # 유저와 팀 텍스트 구성 (스킬 + 지역 + 목표)
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

    for team, sim_score in zip(teams, similarities):
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        skill_match_ratio = len(set(user["skills"]) & set(team_skills)) / max(len(team_skills), 1)

        # ✅ 점수 계산 (유사도 + 스킬일치율)
        # 유사도 비중 85%, 스킬 일치 비중 15%로 약간 더 높게 설정
        score = round(0.9 * sim_score + 0.1 * skill_match_ratio, 2)

        results.append({
            "team_id": team["team_id"],
            "score": score,
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
