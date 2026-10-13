"""
Problem 3: Tilt angle from a gyroscope and an accelerometer.
Run from the repository root:  python starter/problem3_starter.py

Inputs you may use: gyro_rate (deg/s) and accel_angle (deg). The true_angle column
is only for plotting and error. Angles are in degrees.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "problem3_tilt.csv")

dt = 0.01             # time step in seconds (100 Hz)
ACC_STD = 3.0         # accelerometer angle noise (std, degrees)

gyro = df["gyro_rate"].to_numpy()
acc = df["accel_angle"].to_numpy()
N = len(gyro)

gyro_int = np.full(N, np.nan)     # gyro integrated on its own (drifts)
comp = np.full(N, np.nan)         # complementary filter output
kf = np.full(N, np.nan)           # Kalman filter output

# ----------------------------- TODO -----------------------------
# A. Gyro only: integrate gyro_rate over time, starting from acc[0].
# B. Complementary filter: angle = a * (angle + gyro*dt) + (1 - a) * acc, with a around 0.98.
# C. Kalman filter with state [angle, gyro_bias]. The gyro is the control input in the
#    predict step; the accelerometer angle is the measurement in the update step.
for k in range(N):
    pass
# -----------------------------------------------------------------

def rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

ta = df.true_angle.to_numpy()
for name, arr in [("accelerometer", acc), ("gyro integrated", gyro_int),
                  ("complementary", comp), ("kalman", kf)]:
    print(f"Angle RMSE, {name:16s}:", round(rmse(arr, ta), 3))

plt.figure(figsize=(11, 6))
plt.plot(df.t, acc, ".", ms=2, alpha=0.3, label="accelerometer angle")
plt.plot(df.t, gyro_int, label="gyro integrated")
plt.plot(df.t, comp, label="complementary")
plt.plot(df.t, kf, "r", label="kalman")
plt.plot(df.t, ta, "k--", label="true")
plt.ylim(-60, 60); plt.xlabel("time (s)"); plt.ylabel("angle (deg)"); plt.legend()
plt.tight_layout(); plt.savefig("problem3_result.png", dpi=120); plt.show()
