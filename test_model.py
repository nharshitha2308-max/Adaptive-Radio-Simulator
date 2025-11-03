import joblib
import pandas as pd

# Load model and encoder
model = joblib.load('modulation_model_new.joblib')
encoder = joblib.load('channel_encoder.joblib')

# Prepare the test input
channel_encoded = encoder.transform(pd.DataFrame(['Rician'], columns=['Channel_Type']))
channel_encoded_df = pd.DataFrame(channel_encoded, columns=encoder.get_feature_names_out())

# Make 3 example samples
test = pd.DataFrame({
    'SNR_dB': [20, 25, 30],
    'Signal_Power_dBm': [10, 10, 10],
    'Bandwidth_kHz': [25, 25, 25],
    'Latency_ms': [50, 50, 50]
})

# Combine with encoded channel columns
test = pd.concat([test, pd.concat([channel_encoded_df]*3, ignore_index=True)], axis=1)

# Predict
print(model.predict(test))