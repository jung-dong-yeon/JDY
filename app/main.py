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

@app.get("/")
def read_root():
    return {"message": "FastAPI 벡터화 기반 추천 서버 실행 중!"}

# 요청 데이터 모델
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

# 추천 API
@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    return get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])
