import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 벡터라이저 로드
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

def get_recommended_teams(user: dict, teams: list):
    results = []

    # 유저 정보 텍스트 생성
    user_text = " ".join(user["skills"] + [user["region"], user["target"]])

    for team in teams:
        # 팀 정보 텍스트 생성
        team_skills = [s.strip() for s in team["recruitment_skill"].split(",") if s.strip()]
        team_text = " ".join(team_skills + [team["region"], team["goal"]])

        # 벡터화 및 유사도 계산
        vectors = vectorizer.transform([user_text, team_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        # 스킬 일치율 계산
        skill_match_ratio = len(set(user["skills"]) & set(team_skills)) / max(len(team_skills), 1)

        # 가중치 기반 최종 점수 계산
        score = round(0.8 * similarity + 0.2 * skill_match_ratio, 2)

        results.append({
            "team_id": team["team_id"],
            "score": score,
            "prediction": 1 if score >= 0.5 else 0,
            "badge": "추천" if score >= 0.6 else ""
        })

    # 점수 기준 내림차순 정렬
    return sorted(results, key=lambda x: x["score"], reverse=True)
