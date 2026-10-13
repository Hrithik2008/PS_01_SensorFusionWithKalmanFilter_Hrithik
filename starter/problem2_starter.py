"""
Problem 2: Altitude from a barometer and an accelerometer.
Run from the repository root:  python starter/problem2_starter.py

Inputs you may use: baro_alt (altitude, metres) and accel_z (vertical acceleration,
m/s^2, gravity already removed). The true_* columns are only for plotting and error.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "problem2_altitude.csv")

dt = 0.05             # time step in seconds
BARO_STD = 1.0        # barometer noise (std, metres)

z = df["baro_alt"].to_numpy()       # measurement
u = df["accel_z"].to_numpy()        # control input (use it in the predict step)
N = len(z)

est_alt = np.full(N, np.nan)
est_vel = np.full(N, np.nan)

# ----------------------------- TODO -----------------------------
# State: [altitude, vertical velocity]. The accelerometer drives the PREDICT step
# (x = F x + B u); the barometer is the measurement in the UPDATE step.
# Define F, B, H, Q, R, the initial x and P, then fill est_alt[k] and est_vel[k].

a = 0.0008

x = np.array([[0], [0]])
A = np.array([[1, dt], [0, 1]])
B = np.array([[1/2*dt**2], [dt]])
H = np.array([[1, 0]])
Q = np.array([[a, 0], [0, a]])
R = np.array([[BARO_STD ** 2]])
P = np.array([[1, 0], [0, 1]])

for k in range(N):
    x_pred = A @ x + B * u[k]
    P_pred = A @ P @ A.T + Q

    kalman_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + kalman_gain @ (z[k] - H @ x_pred)
    P = (np.eye(2) - kalman_gain @ H) @ P_pred

    est_alt[k] = x[0, 0]
    est_vel[k] = x[1, 0]
# -----------------------------------------------------------------

# Accelerometer alone, for comparison: integrate twice.
vel_acc = np.cumsum(u) * dt
alt_acc = np.cumsum(vel_acc) * dt

def rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

ta = df.true_alt.to_numpy()
print("Altitude RMSE, barometer only   :", round(rmse(z, ta), 3))
print("Altitude RMSE, accelerometer only:", round(rmse(alt_acc, ta), 3))
print("Altitude RMSE, your filter      :", round(rmse(est_alt, ta), 3))

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(df.t, z, ".", ms=3, alpha=0.4, label="barometer")
ax[0].plot(df.t, alt_acc, label="accelerometer only (double integration)")
ax[0].plot(df.t, ta, "k", label="true")
ax[0].plot(df.t, est_alt, "r", label="your estimate")
ax[0].set_ylim(-5, 20); ax[0].set_ylabel("altitude (m)"); ax[0].legend()
ax[1].plot(df.t, df.true_vel, "k", label="true")
ax[1].plot(df.t, est_vel, "r", label="estimate")
ax[1].set_ylabel("velocity (m/s)"); ax[1].set_xlabel("time (s)"); ax[1].legend()
plt.tight_layout(); plt.savefig("problem2_result.png", dpi=120); plt.show()
