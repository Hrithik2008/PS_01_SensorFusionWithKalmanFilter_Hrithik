"""
Problem 4: 2D drone tracking with GPS and an accelerometer.
Run from the repository root:  python starter/problem4_starter.py

Inputs you may use: gps_x, gps_y (metres, only valid when gps_valid == 1, otherwise empty/NaN)
and ax, ay (acceleration in the world frame, m/s^2). The true_* columns are for plotting and error.
The GPS arrives at 2 Hz but the rows come at 50 Hz: predict every row, update only when gps_valid == 1.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "problem4_gps_2d.csv")

dt = 0.02             # time step in seconds (50 Hz)
GPS_STD = 2.5         # GPS noise per axis (std, metres)
uncertainty = 25   # initial uncertainty for the state
gps = df[["gps_x", "gps_y"]].to_numpy()
valid = df["gps_valid"].to_numpy()
u = df[["ax", "ay"]].to_numpy() 
N = len(df)

est = np.full((N, 2), np.nan)
q_hist = []

# ----------------------------- TODO -----------------------------
# State: [x, y, vx, vy]. Define F, B (maps [ax, ay] into the state), H (picks x and y),
# Q, R, the initial x and P. In the loop: predict every row, update only if valid[k] == 1.

def kalman_filter(sa,uncertainty,window=20,scale_lim=(0.01, 10),eta=0.3):
    global est,q_hist
    q = 1
    first = np.flatnonzero(valid == 1)[0]
    x = np.zeros((4, 1))
    x[:2, 0] = gps[first]
    P = np.diag([GPS_STD**2, GPS_STD**2, uncertainty, uncertainty])

    nis_buffer = deque(maxlen=window)

    A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

    B = np.array([[1/2*dt**2, 0],
              [0, 1/2*dt**2],
              [dt, 0],
              [0, dt]])

    H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
    Q = B @ (sa**2 * np.eye(2)) @ B.T

    R = np.array([[GPS_STD ** 2, 0],
                [0, GPS_STD ** 2]])
    gps_count = 0
    for k in range(N):

        x_pred = A @ x + B @ u[k].reshape(-1, 1)
        P_pred = A @ P @ A.T + q * Q

        if valid[k] == 1:

            innovation = gps[k].reshape(-1, 1) - H @ x_pred
            innovation_cov = H @ P_pred @ H.T + R            

            nis = (innovation.T @ np.linalg.inv(innovation_cov) @ innovation).item()

            gps_count += 1
            if gps_count > 10:                    
                nis_buffer.append(nis)
            if len(nis_buffer) == window:     
                mean_nis = np.mean(nis_buffer)
                #print(f"mean_nis: {mean_nis:.3f}, q: {q:.3f}")
                r = mean_nis / 2
                q = float(np.clip(q * r**eta, 0.1, 10))
            kalman_gain = P_pred @ H.T @ np.linalg.inv(innovation_cov)
            x = x_pred + kalman_gain @ innovation
            P = (np.eye(4) - kalman_gain @ H) @ P_pred
        else:
            x = x_pred
            P = P_pred
        q_hist.append(q)
        est[k] = x[:2, 0]

sa = .9

kalman_filter(sa,uncertainty,window=20,scale_lim=(0.01, 10),eta=0.1)

#rint(q_hist[-1])

tx, ty = df.true_x.to_numpy(), df.true_y.to_numpy()
rmse_gps = np.sqrt(np.nanmean((gps[:, 0] - tx) ** 2 + (gps[:, 1] - ty) ** 2))
rmse_est = np.sqrt(np.nanmean((est[:, 0] - tx) ** 2 + (est[:, 1] - ty) ** 2))
print("kalman_filter(sa):", sa , "uncertainty:", uncertainty)
print("2D position RMSE, GPS only  :", round(float(rmse_gps), 3))
print("2D position RMSE, your filter:", round(float(rmse_est), 3))

plt.figure(figsize=(9, 6))
plt.plot(tx, ty, "k", label="true path")
plt.plot(gps[:, 0], gps[:, 1], "o", ms=4, alpha=0.5, label="GPS")
plt.plot(est[:, 0], est[:, 1], "r", label="your estimate")
plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.legend()
plt.tight_layout(); plt.savefig("problem4_result.png", dpi=120); plt.show()
