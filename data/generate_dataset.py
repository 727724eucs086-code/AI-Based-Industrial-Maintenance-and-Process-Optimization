"""
Generates a dataset that replicates the AI4I 2020 Predictive Maintenance
Dataset (Matzka, 2020): same 14 columns, same generative idea (five
independent failure modes combining into one binary "Machine failure"
label), calibrated to land close to the real dataset's ~3.4% overall
failure rate and its per-mode counts (TWF=46, HDF=115, PWF=95, OSF=98,
RNF=9 in the original 10,000-row release).

If you have the actual UCI file, you can drop it in as data/ai4i2020.csv
instead - the schema is identical, so nothing else in the project needs
to change.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# --- Type / quality variant ------------------------------------------------
types = np.random.choice(["L", "M", "H"], size=N, p=[0.5, 0.3, 0.2])
counters = {"L": 0, "M": 0, "H": 0}
product_ids = []
for t in types:
    counters[t] += 1
    product_ids.append(f"{t}{counters[t]:05d}")

# --- Air / process temperature ---------------------------------------------
air_temp = np.random.normal(300, 2, N)
process_temp = air_temp + 10 + np.random.normal(0, 1, N)

# --- Torque & rotational speed (independent, realistic spread) -------------
torque = np.clip(np.random.normal(40, 6, N), 12, 68)
rot_speed = np.clip(np.random.normal(1538, 100, N), 1168, 2886).round().astype(int)
power_actual = torque * (rot_speed * 2 * np.pi / 60)

# --- Tool wear (increments per process by quality variant, resets on
#     replacement) ----------------------------------------------------------
wear_increment = {"L": 2, "M": 3, "H": 5}
tool_wear = np.zeros(N, dtype=int)
twf = np.zeros(N, dtype=int)
current = 0
replace_threshold = np.random.randint(200, 241)
for i, t in enumerate(types):
    current += wear_increment[t]
    tool_wear[i] = current
    if current >= replace_threshold:
        if np.random.rand() < 0.20:  # tool fails in service before swap
            twf[i] = 1
        current = 0
        replace_threshold = np.random.randint(200, 241)

# --- Failure modes -----------------------------------------------------------
hdf = (((process_temp - air_temp) < 8.6) & (rot_speed < 1380)).astype(int)
pwf = ((power_actual < 3500) | (power_actual > 9000)).astype(int)

osf_threshold = np.select([types == "L", types == "M", types == "H"], [11000, 12000, 13000])
osf = ((tool_wear * torque) > osf_threshold).astype(int)

rnf = (np.random.rand(N) < 0.001).astype(int)

machine_failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

df = pd.DataFrame({
    "UDI": np.arange(1, N + 1),
    "Product ID": product_ids,
    "Type": types,
    "Air temperature [K]": air_temp.round(1),
    "Process temperature [K]": process_temp.round(1),
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque.round(1),
    "Tool wear [min]": tool_wear,
    "Machine failure": machine_failure,
    "TWF": twf,
    "HDF": hdf,
    "PWF": pwf,
    "OSF": osf,
    "RNF": rnf,
})

df.to_csv("data/ai4i2020.csv", index=False)
print(f"Wrote data/ai4i2020.csv  shape={df.shape}")
print("Failure rate: %.2f%%" % (machine_failure.mean() * 100))
print("TWF=%d HDF=%d PWF=%d OSF=%d RNF=%d" % (twf.sum(), hdf.sum(), pwf.sum(), osf.sum(), rnf.sum()))
