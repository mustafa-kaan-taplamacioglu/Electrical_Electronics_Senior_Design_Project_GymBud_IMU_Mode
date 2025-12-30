import time
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt

# ================== USER SETTINGS ==================

# RecGym.csv dosya yolu
CSV_PATH = "RecGym.csv"

# Dataset yapısı:
# Kaggle versiyonunda:  Subject, Position, Session, A_x, A_y, A_z, G_x, G_y, G_z, C_1, Workout
# UCI versiyonunda:     Object,  Workout,  Position,           A_x, A_y, A_z, G_x, G_y, G_z, C_1
#
# Aşağıdaki isimleri senin dosyana göre düzenleyebilirsin:
COL_SUBJECT   = "Subject"   # UCI kullanıyorsan None yap
COL_SESSION   = "Session"   # UCI kullanıyorsan None yap
COL_POSITION  = "Position"
COL_WORKOUT   = "Workout"
COL_AX        = "A_x"
COL_AY        = "A_y"
COL_AZ        = "A_z"
COL_GX        = "G_x"
COL_GY        = "G_y"
COL_GZ        = "G_z"

# Filtreler (None bırakırsan o filtreyi kullanmaz)
FILTER_SUBJECT  = 1          # Örn: sadece Subject 1 -> None yaparsan hepsi
FILTER_SESSION  = 1          # Örn: sadece Session 1 -> None
FILTER_POSITION = "wrist"    # "wrist" / "pocket" / "calf" / None
FILTER_WORKOUTS = None       # Örn: ["Squat", "BenchPress"] veya None

# Kaç satır stream edilsin? (çok büyük diye kısaltmak için)
MAX_ROWS = 20000             # None yaparsan hepsini dener (yavaş ve ağır olabilir)

# RecGym sensör sampling rate
SAMPLE_RATE_HZ = 20.0        # 20 Hz -> 0.05 s aralık
REALTIME_SPEED = 1.0         # 1.0 = gerçek zaman, 0.5 = 2x hızlı, 2.0 = 0.5x

# Grafikte tutulacak maksimum nokta
MAX_POINTS = 500

# ===================================================


def load_recgym_subset():
    """RecGym.csv'den filtrelenmiş bir DataFrame döndürür."""
    print(f"Loading {CSV_PATH} ... (this can take a bit)")
    # low_memory=False: tip tahmini daha sağlam
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Bazı versiyonlarda kolon isimleri biraz farklı olabilir, burada kontrol ediyoruz
    cols = df.columns.tolist()
    required_cols = [COL_POSITION, COL_AX, COL_AY, COL_AZ, COL_GX, COL_GY, COL_GZ]
    for c in required_cols:
        if c not in cols:
            raise ValueError(f"Column '{c}' not found in CSV. Found columns: {cols}")

    if COL_WORKOUT not in cols:
        raise ValueError(f"Column '{COL_WORKOUT}' not found in CSV. Needed for exercises.")

    # Filtreler
    if COL_SUBJECT is not None and COL_SUBJECT in cols and FILTER_SUBJECT is not None:
        df = df[df[COL_SUBJECT] == FILTER_SUBJECT]

    if COL_SESSION is not None and COL_SESSION in cols and FILTER_SESSION is not None:
        df = df[df[COL_SESSION] == FILTER_SESSION]

    if FILTER_POSITION is not None:
        df = df[df[COL_POSITION] == FILTER_POSITION]

    if FILTER_WORKOUTS is not None:
        df = df[df[COL_WORKOUT].isin(FILTER_WORKOUTS)]

    # MAX_ROWS ile kısalt
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    df = df.reset_index(drop=True)
    print(f"Filtered rows: {len(df)}")

    # Egzersiz dağılımını konsola bas
    print("Workout counts:")
    print(df[COL_WORKOUT].value_counts())

    return df


def recgym_realtime_plot(df):
    """DataFrame'den satır satır geçerek 6 ekseni gerçek zamanlı çizer."""
    axes_names = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]
    col_map = {
        "A_x": COL_AX,
        "A_y": COL_AY,
        "A_z": COL_AZ,
        "G_x": COL_GX,
        "G_y": COL_GY,
        "G_z": COL_GZ,
    }

    # Çalışma hızını saniye cinsinden
    dt = (1.0 / SAMPLE_RATE_HZ) / REALTIME_SPEED

    # Workout -> renk map’i
    workouts = sorted(df[COL_WORKOUT].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {w: cmap(i % 10) for i, w in enumerate(workouts)}

    # Her eksen için zaman serisi
    x_data = {axis: deque(maxlen=MAX_POINTS) for axis in axes_names}
    y_data = {axis: deque(maxlen=MAX_POINTS) for axis in axes_names}

    # Her eksen için aktif line objesi (tek line, rengi aktif workout'a göre değişiyor)
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs = axs.flatten()

    lines = {}
    for i, axis_name in enumerate(axes_names):
        ax_plot = axs[i]
        line, = ax_plot.plot([], [], lw=1.5)
        lines[axis_name] = line

        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel(axis_name)
        ax_plot.grid(True)

    fig.suptitle("RecGym Real-time IMU (filtered subset)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Legend'da egzersizleri ayrı kutu olarak göstermek için:
    # sadece fig düzeyinde bir legend yapıyoruz.
    legend_patches = [
        plt.Line2D([0], [0], color=color_map[w], lw=2, label=w)
        for w in workouts
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4)

    sample_idx = 0
    last_workout = None

    print("Starting RecGym stream... Ctrl+C to stop.")
    try:
        for _, row in df.iterrows():
            workout = row[COL_WORKOUT]

            if workout != last_workout:
                print(f"\n>>> Now workout: {workout}")
                last_workout = workout

            # Zaman indeksi
            t = sample_idx
            sample_idx += 1

            # Her eksen için veriyi ekle
            for axis_name in axes_names:
                col_name = col_map[axis_name]
                val = float(row[col_name])

                x_data[axis_name].append(t)
                y_data[axis_name].append(val)

            # Çizgileri güncelle
            for i, axis_name in enumerate(axes_names):
                ax_plot = axs[i]
                line = lines[axis_name]

                xs = list(x_data[axis_name])
                ys = list(y_data[axis_name])

                # Line rengi o anda bulunan workout'un rengi
                line.set_color(color_map[workout])
                line.set_xdata(xs)
                line.set_ydata(ys)

                if xs:
                    ax_plot.set_xlim(min(xs), max(xs))
                    ymin = min(ys)
                    ymax = max(ys)
                    if ymin == ymax:
                        ymin -= 1.0
                        ymax += 1.0
                    ax_plot.set_ylim(ymin - 0.1 * abs(ymin if ymin != 0 else 1),
                                     ymax + 0.1 * abs(ymax if ymax != 0 else 1))

            fig.suptitle(
                f"RecGym Real-time IMU – Workout: {workout} | "
                f"Position: {row[COL_POSITION]}",
                fontsize=14,
            )

            plt.draw()
            plt.pause(0.001)
            time.sleep(dt)

        print("\nStream finished.")

    except KeyboardInterrupt:
        print("\nStopped by user (KeyboardInterrupt).")

    finally:
        plt.ioff()
        plt.show()


def main():
    df = load_recgym_subset()
    recgym_realtime_plot(df)


if __name__ == "__main__":
    main()
