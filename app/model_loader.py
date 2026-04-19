import joblib

def load_model():
    return joblib.load("models/energy_model.pkl")