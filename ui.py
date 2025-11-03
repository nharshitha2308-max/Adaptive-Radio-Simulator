# ui.py — Adaptive Radio Communication Simulator (Improved BER + Plot Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.special import erfc
from collections import Counter

# --- Page Setup ---
st.set_page_config(page_title="Adaptive Radio Simulator", layout="centered")
st.title("📡 Adaptive Radio Communication Simulator")

# --- Load trained model and encoder ---
@st.cache_resource
def load_model_encoder():
    model = joblib.load('modulation_model_new.joblib')
    encoder = joblib.load('channel_encoder.joblib')
    return model, encoder

model, encoder = load_model_encoder()

# --- AWGN BER Simulator (Realistic) ---
def simulate_awgn(modulation, snr_db):
    """Simulate BER for a given modulation under AWGN conditions."""
    modulation = modulation.upper().replace(" ", "").replace("-", "")
    snr_linear = 10 ** (snr_db / 10)

    if modulation == 'BPSK':
        ber = 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation == 'QPSK':
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    elif modulation == '8PSK':
        ber = erfc(np.sqrt(snr_linear * np.log2(8)) * np.sin(np.pi / 8)) / np.log2(8)
    elif modulation == '16QAM':
        ber = 3 / 8 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == '32QAM':
        ber = 5 / 12 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == '64QAM':
        ber = 7 / 24 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == 'FSK':
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    else:
        st.warning(f"⚠️ Unknown modulation '{modulation}', defaulting to QPSK.")
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    return float(np.mean(ber))

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Inputs")
snr_values_input = st.sidebar.text_input("SNR values (comma-separated)", "-5,0,5,10,15,20,25,30")
signal_power = st.sidebar.number_input("Signal Power (dBm)", value=0.0)
channel_type = st.sidebar.selectbox("Channel Type", ["AWGN", "Rayleigh", "Rician"])
latency = st.sidebar.number_input("Target Latency (ms)", value=50.0)
bandwidth = 25  # fixed kHz

# --- Run Simulation ---
if st.sidebar.button("Run Simulation"):
    try:
        snr_values = [float(s.strip()) for s in snr_values_input.split(',')]
    except:
        st.error("Invalid SNR input! Please enter comma-separated numbers.")
        st.stop()

    pred_mods = []
    for snr in snr_values:
        # Encode channel type
        channel_df = pd.DataFrame([[channel_type]], columns=['Channel_Type'])
        channel_encoded = encoder.transform(channel_df)
        channel_encoded_df = pd.DataFrame(channel_encoded, columns=encoder.get_feature_names_out())

        # Prepare model input
        df_input = pd.DataFrame([[snr, signal_power, latency]],
                        columns=['SNR_dB', 'Signal_Power_dBm', 'Latency_ms'])
        df_input = pd.concat([df_input.reset_index(drop=True), channel_encoded_df.reset_index(drop=True)], axis=1)

        # Predict modulation
        pred_mod = model.predict(df_input)[0]
        pred_mods.append(pred_mod)

    # --- Choose majority modulation ---
    most_common_mod = Counter(pred_mods).most_common(1)[0][0]
    st.subheader(f"📶 Majority Modulation Chosen: {most_common_mod}")

    # --- Simulate BER for all SNRs using chosen modulation ---
    results = []
    for snr in snr_values:
        ber = simulate_awgn(most_common_mod, snr)
        results.append({'SNR_dB': snr, 'Modulation': most_common_mod, 'BER': ber})

    results_df = pd.DataFrame(results)

    # --- Display Table ---
    st.subheader("📊 Simulation Results")
    st.dataframe(results_df)

    # --- Plot ---
    st.subheader("📈 BER vs SNR")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'd', 'x', '*']
    color = colors[hash(most_common_mod) % len(colors)]
    marker = markers[hash(most_common_mod) % len(markers)]

    ax.semilogy(results_df['SNR_dB'], results_df['BER'],
                marker=marker, linestyle='-', color=color, label=most_common_mod)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"BER vs SNR using Majority Modulation ({most_common_mod})")
    ax.grid(True, which='both')
    ax.legend()
    st.pyplot(fig)

    # --- Save results ---
    results_df.to_csv('user_simulation_results_new.csv', index=False)
    st.success("✅ Results saved as user_simulation_results_new.csv")