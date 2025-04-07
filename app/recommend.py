import pandas as pd
import joblib

model = joblib.load("team_recommender.pkl")

def predict_match_score(user, team):
    test = pd.DataFrame([{
        "skill": team["recruitment_skill"],
        "region": team["region"],
        "target": team["goal"]
    }])
    test_encoded = pd.get_dummies(test)

    for col in model.feature_names_in_:
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    test_encoded = test_encoded[model.feature_names_in_]

    prob = model.predict_proba(test_encoded)[0][1]
    return prob

def get_recommended_teams(user, teams):
    for team in teams:
        team["score"] = predict_match_score(user, team)
    return sorted(teams, key=lambda t: t["score"], reverse=True)
