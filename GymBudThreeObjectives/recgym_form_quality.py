# recgym_form_quality.py
# Real-time style playback of RecGym IMU signals:
# 3 virtual nodes (wrist, pocket, calf) × 6 axes (ax,ay,az,gx,gy,gz)
# + simple form-quality metrics over a sliding window.

import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================= USER SETTINGS =========================

CSV_PATH = "RecGym.csv"   # RecGym dataset
FS = 20.0                 # sampling rate (Hz)
DT = 1.0 / FS

# Real time hızını ayarlamak için (1.0 = gerçek zaman, 5.0 = 5 kat hızlı)
PLAYBACK_SPEED = 5.0

# Plot’ta tutulacak maksimum örnek sayısı
MAX_POINTS = 400

# Form kalitesi için sliding window uzunluğu (saniye)
FORM_WINDOW_SEC = 10.0
FORM_WINDOW_SAMPLES = int(FORM_WINDOW_SEC * FS)

# Kullanacağımız pozisyonlar
POSITIONS = ["wrist", "pocket", "leg"]

# ==============================================================


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


def regularity_index(signal: np.ndarray) -> float:
    """
    Simple regularity metric via autocorrelation peak (0–1).
    Higher = more periodic.
    """
    if signal.size < 10:
        return np.nan

    x = signal - np.mean(signal)
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    ac0 = ac[0]
    if ac0 == 0:
        return np.nan

    # ilk birkaç yüz lag içinde en büyük ikinci pik
    max_lag = min(400, ac.size - 1)
    peak = np.max(ac[1:max_lag])
    return float(peak / ac0)


def main():
    print("Loading RecGym dataset...")
    data_by_pos = load_recgym(CSV_PATH)

    # Deque yapıları: her pozisyon × eksen için sliding buffer
    axes = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]

    x_data = {pos: deque(maxlen=MAX_POINTS) for pos in POSITIONS}
    y_data = {
        (pos, axis): deque(maxlen=MAX_POINTS)
        for pos in POSITIONS for axis in axes
    }

    # Form kalitesi için ayrıca acc_mag tutacağız
    acc_mag_buf = {pos: deque(maxlen=FORM_WINDOW_SAMPLES) for pos in POSITIONS}

    idx_by_pos = {pos: 0 for pos in POSITIONS}
    sample_idx = {pos: 0 for pos in POSITIONS}

    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs = axs.flatten()

    # Her eksen + her pozisyon için line objeleri
    lines = {}
    for i, axis in enumerate(axes):
        ax_plot = axs[i]
        for pos in POSITIONS:
            line, = ax_plot.plot([], [], label=pos)
            lines[(axis, pos)] = line

        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel(axis.lower())
        ax_plot.set_title(f"{axis} – RecGym playback")
        ax_plot.grid(True)
        ax_plot.legend()

    fig.suptitle("RecGym Real-time Playback – Form Quality View", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    running = True
    t0 = time.time()

    try:
        while running:
            loop_start = time.time()
            running = False

            # ---- her pozisyon için bir örnek oku ----
            form_text_parts = []
            for pos in POSITIONS:
                df_pos = data_by_pos[pos]
                i_row = idx_by_pos[pos]
                if i_row >= len(df_pos):
                    continue  # bu pozisyonun verisi bitti
                running = True

                row = df_pos.iloc[i_row]
                idx_by_pos[pos] += 1

                # IMU değerleri
                ax_val = float(row["A_x"])
                ay_val = float(row["A_y"])
                az_val = float(row["A_z"])
                gx_val = float(row["G_x"])
                gy_val = float(row["G_y"])
                gz_val = float(row["G_z"])

                s_idx = sample_idx[pos]
                sample_idx[pos] += 1
                x_data[pos].append(s_idx)

                values = {
                    "A_x": ax_val,
                    "A_y": ay_val,
                    "A_z": az_val,
                    "G_x": gx_val,
                    "G_y": gy_val,
                    "G_z": gz_val,
                }

                for axis in axes:
                    y_data[(pos, axis)].append(values[axis])

                # form quality için acc_mag
                acc_mag = np.sqrt(ax_val**2 + ay_val**2 + az_val**2)
                acc_mag_buf[pos].append(acc_mag)

                # bu pozisyondaki son pencere için basit metrikler
                acc_arr = np.array(acc_mag_buf[pos])
                if acc_arr.size > 10:
                    var_acc = np.var(acc_arr)
                    p2p = acc_arr.max() - acc_arr.min()
                    reg_idx = regularity_index(acc_arr)
                    form_text_parts.append(
                        f"{pos}: var={var_acc:.3f}, ptp={p2p:.3f}, reg={reg_idx:.2f}, "
                        f"workout={row['Workout']}"
                    )

            if not running:
                break

            # ---- çizgileri güncelle ----
            for i, axis in enumerate(axes):
                ax_plot = axs[i]
                all_y = []
                any_x = []

                for pos in POSITIONS:
                    xs = list(x_data[pos])
                    ys = list(y_data[(pos, axis)])
                    line = lines[(axis, pos)]
                    line.set_xdata(xs)
                    line.set_ydata(ys)
                    any_x.extend(xs)
                    all_y.extend(ys)

                if any_x and all_y:
                    ax_plot.set_xlim(min(any_x), max(any_x))
                    ymin = min(all_y)
                    ymax = max(all_y)
                    if ymin == ymax:
                        ymin -= 1.0
                        ymax += 1.0
                    ax_plot.set_ylim(ymin, ymax)

            # başlıkta anlık form-quality bilgisi
            if form_text_parts:
                fig.suptitle(
                    "RecGym Real-time Playback – Form Quality\n" +
                    " | ".join(form_text_parts),
                    fontsize=11
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
