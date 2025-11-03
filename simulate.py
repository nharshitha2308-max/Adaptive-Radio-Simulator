import numpy as np
from scipy.special import erfc

def simulate_awgn(modulation, snr_db, num_bits=1000):
    bits = np.random.randint(0, 2, num_bits)

    # --- ✅ Normalize modulation name (fixes 8-PSK, 16 QAM, etc.) ---
    modulation = modulation.upper().replace(" ", "").replace("-", "")

    snr_linear = 10 ** (snr_db / 10)

    # --- ✅ Simulate based on modulation ---
    if modulation == 'BPSK':
        # BPSK
        ber = 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation == 'QPSK':
        # QPSK
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    elif modulation == '8PSK':
        # 8-PSK
        ber = erfc(np.sqrt(snr_linear * np.log2(8)) * np.sin(np.pi / 8)) / np.log2(8)
    elif modulation == '16QAM':
        # 16-QAM
        ber = 3 / 8 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == '32QAM':
        # 32-QAM
        ber = 5 / 12 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == '64QAM':
        # 64-QAM
        ber = 7 / 24 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == 'FSK':
        # FSK
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    else:
        # Default case (avoid crash)
        print(f"⚠️ Unknown modulation '{modulation}', defaulting to QPSK.")
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))

    return float(np.mean(ber))


# --- Optional: Quick standalone test ---
if __name__ == "__main__":
    for mod in ['BPSK', 'QPSK', '8-PSK', '16 QAM', '32-QAM', 'FSK']:
        for snr in [-5, 0, 5, 10, 15]:
            print(f"{mod} @ {snr} dB → BER = {simulate_awgn(mod, snr):.6f}")