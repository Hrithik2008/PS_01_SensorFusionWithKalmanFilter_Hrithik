"""
Problem 5 (bonus): Robustness to missing readings and outliers.
Run from the repository root:  python starter/problem5_starter.py

Pick ONE version by changing VERSION below.
  "1d": problem5a_robust_1d.csv, same setup as Problem 1 (column z_pos, NaN where missing, plus spikes)
  "2d": problem5b_robust_2d.csv, same setup as Problem 4 (GPS lost for several seconds, plus jumps)

Copy your working filter from Problem 1 or 4 into the TODO block, then:
  - skip the update when the reading is missing (NaN, or gps_valid == 0)
  - reject a reading when its innovation is far larger than the filter expects (compare the
    squared innovation divided by its variance S against a threshold, for example 9 in 1D)
"""
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "2d"
root = Path(__file__).resolve().parents[1] / "data"

if VERSION == "1d":
    df = pd.read_csv(root / "problem5a_robust_1d.csv")
    dt, R_STD = 0.1, 3.0
    z = df["z_pos"].to_numpy(); N = len(z)
    est = np.full(N, np.nan)
else:
    df = pd.read_csv(root / "problem5b_robust_2d.csv")
    dt, GPS_STD = 0.02, 2.5
    gps = df[["gps_x", "gps_y"]].to_numpy(); valid = df["gps_valid"].to_numpy()
    u = df[["ax", "ay"]].to_numpy(); N = len(df)
    est = np.full((N, 2), np.nan)
q_hist = []
uncertainty = 25
# ----------------------------- TODO -----------------------------
def kalman_filter(sa,uncertainty,window=20,scale_lim=(0.01, 100),eta=0.3,gate_threshold=13.82):
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
            if nis > gate_threshold: 
                #print(f"Outlier detected at index {k}, NIS: {nis:.3f}")
                x = x_pred
                P = P_pred
            else:
                if gps_count > 10:                   
                    nis_buffer.append(nis)
                if len(nis_buffer) == window:           
                    mean_nis = np.mean(nis_buffer)
                    #print(f"mean_nis: {mean_nis:.3f}, q: {q:.3f}")
                    r = mean_nis / 2
                    q = float(np.clip(q * r**eta, scale_lim[0], scale_lim[1]))
                    nis_buffer.clear()
                kalman_gain = P_pred @ H.T @ np.linalg.inv(innovation_cov)
                x = x_pred + kalman_gain @ innovation
                P = (np.eye(4) - kalman_gain @ H) @ P_pred
        else:
            x = x_pred
            P = P_pred
        q_hist.append(q)
        est[k] = x[:2, 0]

sa = 1.2

kalman_filter(sa,uncertainty,window=15,scale_lim=(0.01, 10),eta=3,gate_threshold=13.82)

# -----------------------------------------------------------------

if VERSION == "1d":
    err_raw = np.sqrt(np.nanmean((z - df.true_pos) ** 2))
    err_est = np.sqrt(np.nanmean((est - df.true_pos.to_numpy()) ** 2))
    plt.plot(df.t, z, ".", ms=4, alpha=0.5, label="measured (with gaps and spikes)")
    plt.plot(df.t, df.true_pos, "k", label="true")
    plt.plot(df.t, est, "r", label="your estimate")
    plt.xlabel("time (s)"); plt.ylabel("position (m)")
else:
    tx, ty = df.true_x.to_numpy(), df.true_y.to_numpy()
    err_raw = np.sqrt(np.nanmean((gps[:, 0] - tx) ** 2 + (gps[:, 1] - ty) ** 2))
    err_est = np.sqrt(np.nanmean((est[:, 0] - tx) ** 2 + (est[:, 1] - ty) ** 2))
    plt.plot(tx, ty, "k", label="true path")
    plt.plot(gps[:, 0], gps[:, 1], "o", ms=4, alpha=0.5, label="GPS (with jumps)")
    plt.plot(est[:, 0], est[:, 1], "r", label="your estimate")
    plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
print("RMSE, raw sensor :", round(float(err_raw), 3))
print("RMSE, your filter:", round(float(err_est), 3))
plt.legend(); plt.tight_layout(); plt.savefig("problem5_result.png", dpi=120); plt.show()
