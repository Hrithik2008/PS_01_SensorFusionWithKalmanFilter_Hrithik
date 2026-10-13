from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "problem1_1d_tracking.csv")

dt = 0.1          # time step in seconds
R_STD = 3.0       # position sensor noise (std, metres), from the sensor "datasheet"

z = df["z_pos"].to_numpy()          # noisy position measurements (the only input you may use)
N = len(z)

est_pos = np.full(N, np.nan)        # fill these two arrays with your filter output
est_vel = np.full(N, np.nan)

a = 0.001

x = np.array([[0], [0]])
A = np.array([[1, dt], [0, 1]])
B = np.array([[1/2*dt**2], [dt]])
H = np.array([[1, 0]])
Q = np.array([[a, 0], [0, a]])
R = np.array([[R_STD ** 2]])
P = np.array([[1, 0], [0, 1]])

for k in range(N):

    x_pred = A @ x + B * 0  #u_k = 0 as there is no acceleration
    P_pred = A @ P @ A.T + Q

    kalman_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + kalman_gain @ (z[k] - H @ x_pred)
    P = (np.eye(2) - kalman_gain @ H) @ P_pred

    est_pos[k] = x[0, 0]
    est_vel[k] = x[1, 0]




def rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

print("Position RMSE, raw measurement :", round(rmse(z, df.true_pos.to_numpy()), 3))
print("Position RMSE, your filter     :", round(rmse(est_pos, df.true_pos.to_numpy()), 3))
print("Velocity RMSE, your filter     :", round(rmse(est_vel, df.true_vel.to_numpy()), 3))

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(df.t, z, ".", ms=3, alpha=0.5, label="measured")
ax[0].plot(df.t, df.true_pos, "k", label="true")
ax[0].plot(df.t, est_pos, "r", label="estimate")
ax[0].set_ylabel("position (m)"); ax[0].legend()
ax[1].plot(df.t, df.true_vel, "k", label="true")
ax[1].plot(df.t, est_vel, "r", label="estimate")
ax[1].set_ylabel("velocity (m/s)"); ax[1].set_xlabel("time (s)"); ax[1].legend()
plt.tight_layout(); plt.savefig("problem1_result.png", dpi=120); plt.show()
