from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 루트 확인용 기본 경로 추가
@app.get("/")
def read_root():
    return {"message": "FastAPI 서버 동작 중!"}

# ✅ 모델 정의
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

# ✅ 추천 API 엔드포인트
@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    recommended = get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])
    return recommended
