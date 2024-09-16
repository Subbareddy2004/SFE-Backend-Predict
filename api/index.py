from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.graph_objects as go
import os

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Update the file paths
MODEL_PATH = os.path.join(current_dir, 'rf_model_updated.pkl')
ENCODERS_PATH = os.path.join(current_dir, 'label_encoders.pkl')
DATA_PATH = os.path.join(current_dir, 'updated_match_details_with_impact_scores.csv')

# Load the trained model and encoders
def load_model_and_encoders():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(ENCODERS_PATH, 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

rf_model, encoders = load_model_and_encoders()

# Load your dataset
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

def predict_outcome(batting_team, bowling_team, venue, target, current_score, wickets_left, balls_left):
    runs_left = target - current_score
    balls_consumed = 120 - balls_left
    crr = current_score / (balls_consumed / 6) if balls_consumed > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float('inf')

    batting_impact = df[df['batting_team'] == batting_team]['Batting_Team_Impact_Score'].mean()
    bowling_impact = df[df['bowling_team'] == bowling_team]['Bowling_Team_Impact_Score'].mean()

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

    # Predict probabilities
    probability = rf_model.predict_proba(input_data)[0]
    return probability[1], probability[0]  # Assuming 1 is for batting team win, 0 for bowling team win

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

    batting_impact = df[df['batting_team'] == batting_team]['Batting_Team_Impact_Score'].mean()
    bowling_impact = df[df['bowling_team'] == bowling_team]['Bowling_Team_Impact_Score'].mean()

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
    teams = df['batting_team'].unique().tolist()
    return jsonify(teams)

@app.route('/venues', methods=['GET'])
def get_venues():
    venues = df['city'].unique().tolist()
    return jsonify(venues)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Cricket Prediction API!"})

# Remove the if __name__ == '__main__': block

# Add this line at the end of the file
app = app