# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8085"],  # 허용할 도메인 주소
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 허용할 메서드만 지정
    allow_headers=["*"],
)

# 유저 프로필을 위한 Pydantic 모델
class User(BaseModel):
    skills: List[str]
    region: str
    target: str

# 팀 정보를 위한 Pydantic 모델
class Team(BaseModel):
    team_id: int
    recruitment_skill: str
    region: str
    goal: str

# 추천 요청을 위한 모델
class RecommendRequest(BaseModel):
    user: User
    teams: List[Team]

# 추천 API 엔드포인트
@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    recommended = get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])
    return recommended
