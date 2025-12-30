# recgym_smoothness.py
# Real-time style similarity between current IMU samples and
# dynamically updated per-workout average templates.
#
# 3 virtual nodes (wrist, pocket, leg) × 6 axes (A_x..G_z)
# For each sample:
#   - update running mean for [Position, Workout]
#   - show current vs. average signals on the same plots
#   - compute cosine similarity (%) between current 6D vector
#     and that workout's running-average 6D template.

import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================= USER SETTINGS =========================

CSV_PATH = "RecGym.csv"
FS = 20.0          # dataset sampling rate (Hz)
DT = 1.0 / FS

PLAYBACK_SPEED = 5.0   # 1.0 = gerçek zaman, 5.0 = 5 kat hızlı
MAX_POINTS = 400       # grafikte tutulacak örnek sayısı

POSITIONS = ["wrist", "pocket", "leg"]
AXES = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]

# ================================================================


def load_recgym(csv_path: str) -> dict:
    """Load RecGym.csv and split into dict by Position."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    df["Position_norm"] = df["Position"].str.lower().str.strip()

    data_by_pos = {}
    for pos in POSITIONS:
        sub = df[df["Position_norm"] == pos].reset_index(drop=True)
        if sub.empty:
            print(f"WARNING: no rows for position '{pos}' in dataset.")
        data_by_pos[pos] = sub
    return data_by_pos


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1,1]."""
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return num / den


def main():
    print("Loading RecGym dataset...")
    data_by_pos = load_recgym(CSV_PATH)

    # --- plotting buffers: current signal ---
    x_curr = {pos: deque(maxlen=MAX_POINTS) for pos in POSITIONS}
    y_curr = {
        (pos, axis): deque(maxlen=MAX_POINTS)
        for pos in POSITIONS for axis in AXES
    }

    # --- plotting buffers: running-mean (template) signal ---
    y_avg = {
        (pos, axis): deque(maxlen=MAX_POINTS)
        for pos in POSITIONS for axis in AXES
    }

    # --- running mean templates per (position, workout) ---
    # template_mean[pos][workout] -> np.array([Ax,Ay,Az,Gx,Gy,Gz])
    template_mean = {pos: {} for pos in POSITIONS}
    template_count = {pos: {} for pos in POSITIONS}

    idx_by_pos = {pos: 0 for pos in POSITIONS}
    sample_idx = {pos: 0 for pos in POSITIONS}

    # ------------- plotting setup -------------
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs = axs.flatten()

    # one line for current, one for average per position & axis
    lines_curr = {}  # key: (axis, pos) -> Line2D
    lines_avg = {}

    for i, axis in enumerate(AXES):
        ax_plot = axs[i]
        for pos in POSITIONS:
            # current signal line
            lc, = ax_plot.plot([], [], label=f"{pos} curr")
            # average template line
            la, = ax_plot.plot([], [], linestyle="--", label=f"{pos} avg")
            lines_curr[(axis, pos)] = lc
            lines_avg[(axis, pos)] = la

        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel(axis.lower())
        ax_plot.set_title(f"{axis} – current vs. running-average")
        ax_plot.grid(True)
        ax_plot.legend(fontsize=7)

    fig.suptitle("RecGym Real-time Playback – Similarity to Average Templates",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    running = True

    try:
        while running:
            loop_start = time.time()
            running = False

            similarity_msgs = []

            # -------- her pozisyon için bir örnek oku --------
            for pos in POSITIONS:
                df_pos = data_by_pos[pos]
                i_row = idx_by_pos[pos]
                if i_row >= len(df_pos):
                    continue  # bu pozisyonun verisi bitti
                running = True

                row = df_pos.iloc[i_row]
                idx_by_pos[pos] += 1

                workout = str(row["Workout"])
                # 6D IMU vector
                vec = np.array([
                    float(row["A_x"]),
                    float(row["A_y"]),
                    float(row["A_z"]),
                    float(row["G_x"]),
                    float(row["G_y"]),
                    float(row["G_z"]),
                ])

                # ---- running mean template update ----
                tmpl_mean_pos = template_mean[pos]
                tmpl_cnt_pos = template_count[pos]

                old_mean = tmpl_mean_pos.get(workout)
                old_cnt = tmpl_cnt_pos.get(workout, 0)

                if old_mean is None:
                    new_cnt = 1
                    new_mean = vec.astype(float)
                else:
                    new_cnt = old_cnt + 1
                    # incremental mean update
                    new_mean = old_mean + (vec - old_mean) / new_cnt

                tmpl_mean_pos[workout] = new_mean
                tmpl_cnt_pos[workout] = new_cnt

                # similarity between current vec and running mean
                sim = cosine_similarity(vec, new_mean)
                sim_pct = max(0.0, min(100.0, (sim + 1.0) * 50.0))
                similarity_msgs.append(
                    f"{pos}: {workout} ~ {sim_pct:5.1f}%"
                )

                # --- plotting buffers update ---
                s_idx = sample_idx[pos]
                sample_idx[pos] += 1
                x_curr[pos].append(s_idx)

                axis_values_curr = {
                    "A_x": vec[0],
                    "A_y": vec[1],
                    "A_z": vec[2],
                    "G_x": vec[3],
                    "G_y": vec[4],
                    "G_z": vec[5],
                }
                axis_values_avg = {
                    "A_x": new_mean[0],
                    "A_y": new_mean[1],
                    "A_z": new_mean[2],
                    "G_x": new_mean[3],
                    "G_y": new_mean[4],
                    "G_z": new_mean[5],
                }

                for axis in AXES:
                    y_curr[(pos, axis)].append(axis_values_curr[axis])
                    y_avg[(pos, axis)].append(axis_values_avg[axis])

            if not running:
                break

            # -------- çizgileri güncelle --------
            for i, axis in enumerate(AXES):
                ax_plot = axs[i]
                all_y = []
                all_x = []

                for pos in POSITIONS:
                    xs = list(x_curr[pos])
                    ys_curr = list(y_curr[(pos, axis)])
                    ys_avg = list(y_avg[(pos, axis)])

                    lines_curr[(axis, pos)].set_xdata(xs)
                    lines_curr[(axis, pos)].set_ydata(ys_curr)

                    lines_avg[(axis, pos)].set_xdata(xs)
                    lines_avg[(axis, pos)].set_ydata(ys_avg)

                    all_x.extend(xs)
                    all_y.extend(ys_curr)
                    all_y.extend(ys_avg)

                if all_x and all_y:
                    ax_plot.set_xlim(min(all_x), max(all_x))
                    ymin = min(all_y)
                    ymax = max(all_y)
                    if ymin == ymax:
                        ymin -= 1.0
                        ymax += 1.0
                    ax_plot.set_ylim(ymin, ymax)

            # üst başlıkta benzerlik mesajı
            if similarity_msgs:
                fig.suptitle(
                    "RecGym Real-time Playback – Similarity to Average Templates\n"
                    + " | ".join(similarity_msgs),
                    fontsize=10
                )

            plt.draw()
            plt.pause(0.001)

            # hız kontrolü
            target_dt = DT / PLAYBACK_SPEED
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user (KeyboardInterrupt).")

    print("Playback finished.")


if __name__ == "__main__":
    main()
