import numpy as np
import pandas as pd
import importlib.util
import sys

# -------------------------
# User settings
# -------------------------
module_path = "ieee80211b_tx_dual_mode.py"  # Upload your file in Colab and adjust path if needed
bitstream_length = 400  # number of bits for test vector
data_bits_list = [8, 10, 12, 14, 16]
coeff_bits_list = [8, 10, 12, 14, 16]
acc_bits_list = [20, 24, 28, 32]
csv_filename = "bitwidth_snr_sweep.csv"

# -------------------------
# Load your transmitter module
# -------------------------
spec = importlib.util.spec_from_file_location("txdual", module_path)
txdual = importlib.util.module_from_spec(spec)
sys.modules["txdual"] = txdual
spec.loader.exec_module(txdual)

# -------------------------
# Generate reference floating-point output
# -------------------------
np.random.seed(42)
bits = np.random.randint(0, 2, bitstream_length)

tx_ref = txdual.IEEE80211bTransmitterDualMode(
    sps=8, rrc_span_chips=6, rrc_alpha=0.35,
    data_bits=16, coeff_bits=16, acc_bits=32
)
y_float, _ = tx_ref.transmit(bits, fixed_point=False)

# -------------------------
# Sweep bit widths
# -------------------------
rows = []
for db in data_bits_list:
    for cb in coeff_bits_list:
        for ab in acc_bits_list:
            tx = txdual.IEEE80211bTransmitterDualMode(
                sps=8, rrc_span_chips=6, rrc_alpha=0.35,
                data_bits=db, coeff_bits=cb, acc_bits=ab
            )
            y_fixed, _ = tx.transmit(bits, fixed_point=True)
            snr = txdual.IEEE80211bTransmitterDualMode.snr_db(y_float, y_fixed)
            rows.append({
                "Data bits": db,
                "Coeff bits": cb,
                "Acc bits": ab,
                "SNR (dB)": float(snr)
            })
            print(f"db={db}, cb={cb}, ab={ab} -> SNR={snr:.2f} dB")

# -------------------------
# Save & display results
# -------------------------
df = pd.DataFrame(rows).sort_values(by="SNR (dB)", ascending=False).reset_index(drop=True)
df.to_csv(csv_filename, index=False)

print("\nTop 10 configs by SNR:")
print(df.head(10))

print(f"\nFull sweep saved to: {csv_filename}")
