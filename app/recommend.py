# app/recommend.py

def calculate_match_score(user, team):
    skill_score = 2 if team["recruitment_skill"] in user["skills"] else 0
    region_score = 2 if user["region"] == team["region"] else 0
    target_score = 2 if user["target"] == team["goal"] else 0
    return skill_score + region_score + target_score

def get_recommended_teams(user, teams):
    for team in teams:
        team["score"] = calculate_match_score(user, team)
    return sorted(teams, key=lambda t: t["score"], reverse=True)
