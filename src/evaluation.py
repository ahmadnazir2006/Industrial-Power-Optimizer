import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_sequences(data, window_size=96):

    X, y = [], []

    for i in range(window_size, len(data) - window_size):

        X.append(data[i-window_size:i, :])

        y.append(data[i:i+window_size, 0])

    return np.array(X), np.array(y)


def inverse_transform_96_steps(data_2d, scaler_obj):

    num_samples = data_2d.shape[0]

    final_output = np.zeros((num_samples, 96))

    for t in range(96):

        dummy = np.zeros((num_samples, 7))

        dummy[:, 0] = data_2d[:, t]

        unscaled = scaler_obj.inverse_transform(dummy)

        final_output[:, t] = unscaled[:, 0]

    return final_output


test_scaled = np.load("data/raw/processed/test_scaled.npy")

df = pd.read_csv(
    "data/raw/processed/steel_industry_final.csv",
    parse_dates=['date'],
    index_col='date'
)

X_test_3d, y_test_labels = create_sequences(test_scaled, 96)

split_index = int(len(df) * 0.8)

df_test_dates = df.iloc[
    split_index + 96:
    split_index + 96 + len(y_test_labels)
]

model_scaler_dict = {

    'lstm.h5': "models/lstm_scaler.pkl",

    'cnn_lstm.h5': "models/cnn_lstm_scaler.pkl",

    'gru.h5': "models/gru_scaler.pkl"

}

leaderboard = []

os.makedirs("data/results", exist_ok=True)

for model_file, scaler_path in model_scaler_dict.items():

    print(f"\n🚀 Deep Evaluation: {model_file}")

    model = load_model(f"models/{model_file}")

    scaler = joblib.load(scaler_path)

    y_pred_scaled = model.predict(X_test_3d, verbose=0)

    y_pred_actual = inverse_transform_96_steps(
        y_pred_scaled,
        scaler
    )

    y_test_actual = inverse_transform_96_steps(
        y_test_labels,
        scaler
    )

    mask = y_test_actual > 10.0

    mape = np.mean(
        np.abs(
            (y_test_actual[mask] - y_pred_actual[mask])
            / y_test_actual[mask]
        )
    ) * 100

    accuracy = 100 - mape

    mae_kwh = np.mean(
        np.abs(y_test_actual - y_pred_actual)
    )

    actual_peaks = np.max(y_test_actual, axis=1)

    pred_peaks = np.max(y_pred_actual, axis=1)

    peak_mae = np.mean(
        np.abs(actual_peaks - pred_peaks)
    )

    print(f"📊 Accuracy (Filtered): {accuracy:.2f}%")

    print(f"📊 Avg Error: {mae_kwh:.2f} kWh")

    print(f"📊 Peak Error: {peak_mae:.2f} kWh")

    leaderboard.append({

        'Model': model_file,

        'Accuracy': accuracy,

        'MAE_kWh': mae_kwh,

        'Peak_Error': peak_mae

    })

    sample_index = 500

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(96),
        y_test_actual[sample_index],
        label='Actual 24h Load',
        color='orange',
        linewidth=2.5
    )

    plt.plot(
        range(96),
        y_pred_actual[sample_index],
        label='AI 24h Forecast',
        color='blue',
        linestyle='--',
        linewidth=2
    )

    plt.title(f"24-Hour Optimization Horizon: {model_file}")

    plt.ylabel("Usage (kWh)")

    plt.xlabel("15-Min Intervals")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.show()

    model_tag = model_file.split('.')[0]

    model_results_df = pd.DataFrame({

        'Actual Usage (kWh)': y_test_actual[:, 0],

        'Predicted Usage (kWh)': y_pred_actual[:, 0]

    }, index=df_test_dates.index)

    save_path = f"data/results/{model_tag}_results.csv"

    model_results_df.to_csv(save_path)

    print(f"✅ Dashboard Data saved: {save_path}")

summary_df = pd.DataFrame(leaderboard)

summary_df = summary_df.sort_values(
    by='Accuracy',
    ascending=False
)

print("\n🏆 TOURNAMENT LEADERBOARD 🏆")

print(summary_df)

best_model_name = summary_df.iloc[0]['Model']

best_model_tag = best_model_name.split('.')[0]

best_results_path = f"data/results/{best_model_tag}_results.csv"

print(f"\n✅ Best Model: {best_model_name}")

print(f"✅ Production data ready at: {best_results_path}")