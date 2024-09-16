from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Update the file paths
MODEL_PATH = os.path.join(current_dir, 'rf_model_updated.pkl')
ENCODERS_PATH = os.path.join(current_dir, 'label_encoders.pkl')

# Load the trained model and encoders
def load_model_and_encoders():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(ENCODERS_PATH, 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

rf_model, encoders = load_model_and_encoders()

# Add the data from CSV directly here
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants"
]

VENUES = [
    "Mumbai", "Chennai", "Bangalore", "Kolkata", "Delhi", "Mohali",
    "Jaipur", "Hyderabad", "Ahmedabad", "Lucknow", "Pune", "Dharamsala",
    "Visakhapatnam", "Indore", "Raipur", "Ranchi", "Cuttack", "Kanpur", "Rajkot"
]

# Add average impact scores for each team (calculate these from your CSV)
TEAM_IMPACT_SCORES = {
    "Mumbai Indians": {"batting": 10861.937, "bowling": 11012.365},
    "Chennai Super Kings": {"batting": 10500.123, "bowling": 10800.456},
    "Royal Challengers Bangalore": {"batting": 10750.789, "bowling": 10600.234},
    "Kolkata Knight Riders": {"batting": 10300.567, "bowling": 10400.789},
    "Delhi Capitals": {"batting": 10450.234, "bowling": 10550.678},
    "Punjab Kings": {"batting": 10200.901, "bowling": 10150.345},
    "Rajasthan Royals": {"batting": 10350.678, "bowling": 10250.901},
    "Sunrisers Hyderabad": {"batting": 10150.345, "bowling": 10350.567},
    "Gujarat Titans": {"batting": 10550.123, "bowling": 10650.789},
    "Lucknow Super Giants": {"batting": 10400.456, "bowling": 10500.123}
}

def predict_outcome(batting_team, bowling_team, venue, target, current_score, wickets_left, balls_left):
    runs_left = target - current_score
    balls_consumed = 120 - balls_left
    crr = current_score / (balls_consumed / 6) if balls_consumed > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float('inf')

    batting_impact = TEAM_IMPACT_SCORES[batting_team]["batting"]
    bowling_impact = TEAM_IMPACT_SCORES[bowling_team]["bowling"]

    input_data = pd.DataFrame({
        'batting_team': [encoders['batting_team'].transform([batting_team])[0]],
        'bowling_team': [encoders['bowling_team'].transform([bowling_team])[0]],
        'city': [encoders['city'].transform([venue])[0]],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_left],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'Batting_Team_Impact_Score': [batting_impact],
        'Bowling_Team_Impact_Score': [bowling_impact]
    })

    probability = rf_model.predict_proba(input_data)[0]
    return probability[1], probability[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    venue = data['venue']
    target = data['target']
    current_score = data['current_score']
    wickets_left = data['wickets_left']
    balls_left = data['balls_left']

    batting_prob, bowling_prob = predict_outcome(
        batting_team, bowling_team, venue, target, current_score, wickets_left, balls_left
    )

    batting_impact = TEAM_IMPACT_SCORES[batting_team]["batting"]
    bowling_impact = TEAM_IMPACT_SCORES[bowling_team]["bowling"]

    response = {
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'batting_prob': float(batting_prob),
        'bowling_prob': float(bowling_prob),
        'batting_impact': float(batting_impact),
        'bowling_impact': float(bowling_impact)
    }

    return jsonify(response)

@app.route('/teams', methods=['GET'])
def get_teams():
    return jsonify(TEAMS)

@app.route('/venues', methods=['GET'])
def get_venues():
    return jsonify(VENUES)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Cricket Prediction API!"})

# Remove the if __name__ == '__main__': block

# Add this line at the end of the file
if __name__ == '__main__':
    app.run(debug=True)