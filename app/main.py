from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 정의
class User(BaseModel):
    userSkill: str  # 문자열 형태의 JSON (예: '["AI", "백엔드"]')
    userRegion: str
    userTarget: str

class Team(BaseModel):
    team_id: int
    recruitment_skill: str
    region: str
    goal: str

class RecommendRequest(BaseModel):
    user: User
    teams: List[Team]

# 추천 API
@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    print("🔥 추천 요청 도착")
    print("📥 유저 JSON:", req.user.dict())
    print("📥 팀들 JSON:", [t.dict() for t in req.teams])

    recommended = get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])

    print("🧠 추천 결과:", recommended)
    return recommended
