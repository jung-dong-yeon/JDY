from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .recommend import get_recommended_teams

app = FastAPI()

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
