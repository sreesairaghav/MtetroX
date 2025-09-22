import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pulp

# Paths
DATA_PATH = "data/kmrl_40_trains_history.csv"
SIM_DAY_PATH = "data/simulation_day1.csv"
UPDATED_HISTORY_PATH = "data/updated_history.csv"
MODEL_RF_PATH = "models/rf_classifier.pkl"
MODEL_LE_PATH = "models/label_encoder.pkl"
MODEL_SHUNT_REG_PATH = "models/shunting_regressor.pkl"
DAY1_PRED_STATUS_PATH = "data/day1_assigned_status_prediction.csv"
DAY1_PRED_SHUNTING_PATH = "data/day1_predicted_shunting.csv"
DAY1_SHUNTING_OPT_PATH = "data/day1_shunting_optimization.csv"


def preprocess_data(df):
    df.columns = df.columns.str.strip()

    base_features = ['rs_days_from_plan', 'sig_days_from_plan', 'tel_days_from_plan',
                'job_open_count', 'job_critical_count', 'branding_req_hours',
                'mileage_km', 'bogie_wear_index', 'estimated_shunting_mins',
                'iot_temp_avg_c', 'hvac_alert', 'predicted_failure_risk']

    # One-hot encode the categorical cleaning_slot
    if 'cleaning_slot' not in df.columns:
        raise ValueError("Missing column 'cleaning_slot' in dataset")

    df = pd.get_dummies(df, columns=['cleaning_slot'], drop_first=True)

    features = base_features + [col for col in df.columns if col.startswith('cleaning_slot_')]

    missing_cols = [col for col in features + ['assigned_status'] if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing columns in input data: {}".format(missing_cols))

    df = df.copy()
    le = LabelEncoder()
    df['assignedstatus_enc'] = le.fit_transform(df['assigned_status'].str.strip())
    df.fillna(0, inplace=True)

    X = df[features]
    y = df['assignedstatus_enc']
    return X, y, le


def train_rf_classifier():
    print("Training RandomForest assigned status classifier...")
    df = pd.read_csv(DATA_PATH)
    X, y, le = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print(f"Train accuracy: {rf.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")

    joblib.dump(rf, MODEL_RF_PATH)
    joblib.dump(le, MODEL_LE_PATH)
    print(f"Saved RF model to {MODEL_RF_PATH} and label encoder to {MODEL_LE_PATH}")


def train_shunting_regressor():
    print("Training XGBoost regressor for shunting time...")
    df = pd.read_csv(DATA_PATH)

    features = ['rs_days_from_plan', 'sig_days_from_plan', 'tel_days_from_plan',
                'job_open_count', 'job_critical_count', 'branding_req_hours',
                'mileage_km', 'bogie_wear_index',
                'iot_temp_avg_c', 'hvac_alert', 'predicted_failure_risk']

    # cleaning_slot omitted since it's categorical and not used for shunting regression

    missing_cols = [col for col in features + ['estimated_shunting_mins'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data for shunting: {missing_cols}")

    df = df.fillna(0)
    X = df[features]
    y = df['estimated_shunting_mins']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    print(f"Train R^2: {model.score(X_train, y_train):.3f}")
    print(f"Test R^2: {model.score(X_test, y_test):.3f}")

    joblib.dump(model, MODEL_SHUNT_REG_PATH)
    print(f"Saved shunting regressor model to {MODEL_SHUNT_REG_PATH}")


def simulate_day():
    print("Simulating day-1 data based on history...")
    df_history = pd.read_csv(DATA_PATH)
    last_date = pd.to_datetime(df_history['date'].max())
    new_date = last_date + pd.Timedelta(days=1)

    sample_df = df_history.sample(40, random_state=1).copy()
    sample_df['date'] = new_date

    noise_scale = 0.05
    numeric_cols = sample_df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        sample_df[col] = sample_df[col] * (1 + noise_scale * np.random.randn(len(sample_df)))

    # Add missing columns with default values if not present
    essential_cols = ['rs_days_from_plan', 'sig_days_from_plan', 'tel_days_from_plan',
                      'job_open_count', 'job_critical_count', 'branding_req_hours',
                      'mileage_km', 'bogie_wear_index',
                      'iot_temp_avg_c', 'hvac_alert', 'predicted_failure_risk',
                      'estimated_shunting_mins', 'cleaning_slot', 'assigned_status']
    for col in essential_cols:
        if col not in sample_df.columns:
            sample_df[col] = 0 if col != 'assigned_status' and col != 'cleaning_slot' else 'Unknown'

    sample_df['assigned_status'] = 'Unknown'
    sample_df.to_csv(SIM_DAY_PATH, index=False)
    print(f"Simulated day-1 data saved to {SIM_DAY_PATH}")


def predict_assigned_status():
    print("Predicting assigned status for day-1...")
    df = pd.read_csv(SIM_DAY_PATH)

    if 'cleaning_slot' not in df.columns:
        raise ValueError("'cleaning_slot' missing in simulated data")

    df = pd.get_dummies(df, columns=['cleaning_slot'], drop_first=True)

    base_features = ['rs_days_from_plan', 'sig_days_from_plan', 'tel_days_from_plan',
                     'job_open_count', 'job_critical_count', 'branding_req_hours',
                     'mileage_km', 'bogie_wear_index', 'estimated_shunting_mins',
                     'iot_temp_avg_c', 'hvac_alert', 'predicted_failure_risk']

    features = base_features + [col for col in df.columns if col.startswith('cleaning_slot_')]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    rf = joblib.load(MODEL_RF_PATH)
    le = joblib.load(MODEL_LE_PATH)

    X = df[features].fillna(0)
    y_pred = rf.predict(X)
    df['assignedstatus_pred'] = le.inverse_transform(y_pred)
    df.to_csv(DAY1_PRED_STATUS_PATH, index=False)
    print(f"Assigned status predictions saved to {DAY1_PRED_STATUS_PATH}")


def predict_shunting():
    print("Predicting shunting time for day-1...")
    df = pd.read_csv(SIM_DAY_PATH)

    features = ['rs_days_from_plan', 'sig_days_from_plan', 'tel_days_from_plan',
                'job_open_count', 'job_critical_count', 'branding_req_hours',
                'mileage_km', 'bogie_wear_index',
                'iot_temp_avg_c', 'hvac_alert', 'predicted_failure_risk']

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for shunting prediction: {missing}")

    model = joblib.load(MODEL_SHUNT_REG_PATH)
    X = df[features].fillna(0)
    df['predicted_shuntingmins'] = model.predict(X)
    df.to_csv(DAY1_PRED_SHUNTING_PATH, index=False)
    print(f"Shunting time predictions saved to {DAY1_PRED_SHUNTING_PATH}")


def optimize_shunting():
    print("Optimizing shunting assignments...")
    df = pd.read_csv(DAY1_PRED_SHUNTING_PATH)

    if 'stabling_position' not in df.columns:
        raise ValueError("Missing 'stabling_position' in data")

    if 'predicted_shuntingmins' not in df.columns:
        raise ValueError("Missing 'predicted_shuntingmins' in data")

    trains = df.index.tolist()
    bays = df['stabling_position'].unique()

    prob = pulp.LpProblem("Shunting_Optimization", pulp.LpMinimize)
    assign = pulp.LpVariable.dicts("assign",
                                   ((t, b) for t in trains for b in bays),
                                   cat='Binary')

    predicted_times = df['predicted_shuntingmins'].to_dict()

    prob += pulp.lpSum(assign[(t,b)] * predicted_times[t] for t in trains for b in bays)

    for t in trains:
        prob += pulp.lpSum(assign[(t,b)] for b in bays) == 1

    for b in bays:
        prob += pulp.lpSum(assign[(t,b)] for t in trains) <= 1

    prob.solve()
    results = []
    for t in trains:
        for b in bays:
            if pulp.value(assign[(t,b)]) == 1:
                results.append({
                    'train_index': t,
                    'assigned_bay': b,
                    'predicted_shuntingmins': predicted_times[t]
                })

    pd.DataFrame(results).to_csv(DAY1_SHUNTING_OPT_PATH, index=False)
    print(f"Shunting optimization results saved to {DAY1_SHUNTING_OPT_PATH}")


def update_history():
    print("Updating historical data with simulated day...")
    df_hist = pd.read_csv(DATA_PATH)
    df_new = pd.read_csv(SIM_DAY_PATH)

    df_updated = pd.concat([df_hist, df_new], ignore_index=True)
    df_updated.to_csv(UPDATED_HISTORY_PATH, index=False)
    print(f"Updated history saved to {UPDATED_HISTORY_PATH}")


def main():
    train_rf_classifier()
    train_shunting_regressor()
    simulate_day()
    predict_assigned_status()
    predict_shunting()
    optimize_shunting()
    update_history()


if __name__ == "__main__":
    main()
