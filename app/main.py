from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams  # 추천 알고리즘 함수 import

app = FastAPI()  # FastAPI 앱 생성

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8085"],  # 허용할 도메인 주소
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 허용할 메서드만 지정
    allow_headers=["*"],
)

# 사용자 정보 데이터 모델
class User(BaseModel):
    skills: List[str]  # 사용자 보유 기술 (최대 3개)
    region: str        # 선호 지역
    target: str        # 목표 (예: 1등, 상금, 경험쌓기)

# 팀 정보 데이터 모델
class Team(BaseModel):
    team_id: int
    recruitment_skill: str  # 팀에서 모집하는 기술
    region: str             # 팀 위치
    goal: str               # 팀 목표

# 추천 요청 시 받을 전체 구조
class RecommendRequest(BaseModel):
    user: User         # 사용자 정보
    teams: List[Team]  # 비교할 팀 리스트

# POST 요청 핸들러 (추천 실행)
@app.post("/api/recommend/teams")
def recommend_teams(req: RecommendRequest):
    # 추천 결과 계산
    recommended = get_recommended_teams(req.user.dict(), [t.dict() for t in req.teams])
    return recommended  # 결과 반환
