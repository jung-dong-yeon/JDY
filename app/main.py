from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 🔥 이 줄 추가

from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"] 처럼 도메인 지정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(BaseModel):
    skills: List[str]
    region: str
    target: str

class Team(BaseModel):
    team_id: int
    recruitment_skill: str
    region: str
    goal: str

class RecommendRequest(BaseModel):
    user: User
    teams: List[Team]

@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    recommended = get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])
    return recommended
