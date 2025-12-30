# recgym_metabolic_intensity.py
#
# 1) RecGym.csv dosyasını okur
# 2) acc_mag, ENMO, intensity_score hesaplar
# 3) 6 seviyeli metabolik yoğunluk sınıfı atar (Global Quantile'lar ile)
# 4) Tüm veri için istatistikleri yazar
# 5) Gerçek-zaman benzeri oynatma yapar:
#    - Üst grafik: acc_mag & ENMO
#    - Orta grafik: anlık sınıf (0–5)
#    - Alt grafik: her set için (Null → egzersiz → Null) koşarken
#                  o ana kadar olan sınıf ortalaması (set-mean class)

import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ================== SETTINGS ==================

CSV_PATH = "RecGym.csv"              # RecGym dataset path
OUTPUT_CSV = "RecGym_with_intensity_global_quantile.csv" # Updated output file name

# gerçek-zaman oynatma ayarları
WINDOW_SIZE = 600                    # grafikte tutulacak sample sayısı
STRIDE = 10                          # her adımda kaç örnek atlanacak (1 = hepsi)
PLAYBACK_SLEEP = 0.01               # frame gecikmesi (saniye)

# ==========================================================
# 1) FEATURE HESABI
# ==========================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'e:
      - acc_mag  (IMU büyüklüğü, g)
      - ENMO     (Euclidean Norm Minus One, negatifler 0 yapılmış)
      - intensity_score = acc_mag + 2*ENMO
    kolonlarını ekler.
    """
    # İvme kolon isimleri: gerekirse kendi dosyana göre değiştir
    try:
        ax = df["A_x"].to_numpy()
        ay = df["A_y"].to_numpy()
        az = df["A_z"].to_numpy()
    except KeyError as e:
        print(f"Error: Could not find accelerometer columns (A_x, A_y, A_z) in CSV. Missing: {e}")
        sys.exit(1)

    # Büyüklük
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # ENMO: norm - 1g, negatifleri 0
    enmo = acc_mag - 1.0
    enmo[enmo < 0] = 0.0

    df["acc_mag"] = acc_mag
    df["ENMO"] = enmo

    # Basit intensity skoru (istersen formülü değiştirebilirsin)
    df["intensity_score"] = df["acc_mag"] + 2.0 * df["ENMO"]

    return df


# ==========================================================
# 2) 6 SEVİYELİ YOĞUNLUK SINIFI (GLOBAL QUANTILE BASED)
# ==========================================================

CLASS_LABELS = {
    0: "Class-0: Null",
    1: "Class-1: Low Intensity",
    2: "Class-2: Low-Mid Intensity",
    3: "Class-3: Mid Intensity",
    4: "Class-4: High-Mid Intensity",
    5: "Class-5: High Intensity",
}

def classify_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    intensity_score üzerinden 6 seviyeli sınıf atar.
      - Workout == "Null" → Class-0
      - Diğerleri için intensity_score'un global quantile'larına göre
        Class-1 .. Class-5 arası.
    """
    # Null aktiviteyi ayır
    is_null = df["Workout"] == "Null"

    # Sadece egzersiz örnekleri üzerinden quantile
    scores_ex = df.loc[~is_null, "intensity_score"].to_numpy()

    # Çok büyük dataset, ama quantile hızlıdır
    q20, q40, q60, q80 = np.quantile(scores_ex, [0.2, 0.4, 0.6, 0.8])
    print(f"Global Quantiles for Intensity Score: 20%={q20:.2f}, 40%={q40:.2f}, 60%={q60:.2f}, 80%={q80:.2f}")


    # Başlangıç: hepsini Null sınıfı yap
    cls = np.zeros(len(df), dtype=int)

    # Egzersiz için sınırlar
    s = df["intensity_score"].to_numpy()

    mask1 = (~is_null) & (s <= q20)
    mask2 = (~is_null) & (s >  q20) & (s <= q40)
    mask3 = (~is_null) & (s >  q40) & (s <= q60)
    mask4 = (~is_null) & (s >  q60) & (s <= q80)
    mask5 = (~is_null) & (s >  q80)

    cls[mask1] = 1
    cls[mask2] = 2
    cls[mask3] = 3
    cls[mask4] = 4
    cls[mask5] = 5

    df["intensity_class"] = cls
    df["intensity_label"] = [CLASS_LABELS[c] for c in cls]
    return df


# ==========================================================
# 3) GERÇEK-ZAMAN OYNATMA + SET ORTALAMA SINIF GRAFİĞİ
# ==========================================================

def realtime_playback(df: pd.DataFrame) -> None:
    """
    Tüm RecGym verisini sıralı şekilde oynatır.
    Üst: acc_mag & ENMO
    Orta: anlık intensity_class (0–5)
    Alt: her set (Null → egzersiz → Null) için o ana kadar olan
          sınıf ortalaması (set-mean class).
    """

    # Zaman sırasına koy (subject/position/session zaten ardışık ama garanti olsun)
    sort_cols = [c for c in ["Subject", "Position", "Session"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    acc = df["acc_mag"].to_numpy()
    enmo = df["ENMO"].to_numpy()
    cls = df["intensity_class"].to_numpy()
    workout = df["Workout"].astype(str).to_numpy()

    n = len(df)

    # Deque yapıları
    x_hist = deque(maxlen=WINDOW_SIZE)
    acc_hist = deque(maxlen=WINDOW_SIZE)
    enmo_hist = deque(maxlen=WINDOW_SIZE)
    cls_hist = deque(maxlen=WINDOW_SIZE)
    setmean_hist = deque(maxlen=WINDOW_SIZE)

    # Set takibi
    in_set = False
    set_sum = 0.0
    set_count = 0
    current_set_mean = 0.0

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Üst grafik: acc_mag & ENMO
    line_acc, = ax1.plot([], [], label="acc_mag", color='blue', alpha=0.6)
    line_enmo, = ax1.plot([], [], label="ENMO", color='orange', alpha=0.8)
    ax1.set_ylabel("g")
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # Orta grafik: anlık sınıf
    line_cls, = ax2.plot([], [], drawstyle="steps-post", color='green')
    ax2.set_ylabel("Class")
    ax2.set_ylim(-0.5, 5.5)
    ax2.set_yticks(range(6))
    ax2.grid(True, axis='y')

    # Alt grafik: set ortalama sınıf
    line_setmean, = ax3.plot([], [], drawstyle="steps-post", color='purple', linewidth=2)
    ax3.set_ylabel("Set Mean Class")
    ax3.set_ylim(-0.5, 5.5)
    ax3.set_yticks(range(6))
    ax3.set_xlabel("Sample index")
    ax3.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    try:
        print("Starting real-time playback. Close window or Ctrl+C to stop.")

        for i in range(0, n, STRIDE):
            acc_i = acc[i]
            enmo_i = enmo[i]
            cls_i = int(cls[i])
            w_i = workout[i]

            # ------- Set başlangıç / bitiş mantığı -------
            if w_i == "Null":
                # Egzersiz dışı bölge
                if in_set:
                    # set bitti
                    in_set = False
                    set_sum = 0.0
                    set_count = 0
                current_set_mean = 0.0   # Null döneminde ortalama 0 gösterelim
            else:
                # Egzersiz bölgesi
                if not in_set:
                    # yeni set başlangıcı
                    in_set = True
                    set_sum = cls_i
                    set_count = 1
                else:
                    set_sum += cls_i
                    set_count += 1
                current_set_mean = set_sum / set_count

            # ------- Deque’lere ekle -------
            x_hist.append(i)
            acc_hist.append(acc_i)
            enmo_hist.append(enmo_i)
            cls_hist.append(cls_i)
            setmean_hist.append(current_set_mean)

            # ------- Çizgileri güncelle -------
            xs = list(x_hist)

            line_acc.set_xdata(xs)
            line_acc.set_ydata(list(acc_hist))

            line_enmo.set_xdata(xs)
            line_enmo.set_ydata(list(enmo_hist))

            line_cls.set_xdata(xs)
            line_cls.set_ydata(list(cls_hist))

            line_setmean.set_xdata(xs)
            line_setmean.set_ydata(list(setmean_hist))

            # Dinamik x-limit
            if xs:
                ax1.set_xlim(xs[0], xs[-1])
                # acc/ENMO için otomatik y-limit
                y1 = list(acc_hist) + list(enmo_hist)
                if y1:
                    ymin = min(y1)
                    ymax = max(y1)
                    if ymin == ymax:
                        ymin -= 0.1
                        ymax += 0.1
                    # Y-ekseni aralığına biraz boşluk ekle
                    margin = 0.05 * (ymax - ymin)
                    ax1.set_ylim(ymin - margin, ymax + margin)

            # Başlık
            lbl = CLASS_LABELS[cls_i]
            fig.suptitle(
                f"RecGym Playback (Global Quantile Based)\n" # Updated title
                f"Sample {i+1}/{n}  |  Workout: {w_i}  |  "
                f"{lbl}  |  "
                f"Current Set Mean: {current_set_mean:.2f}",
                fontsize=11
            )

            plt.draw()
            plt.pause(PLAYBACK_SLEEP)

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")

    finally:
        plt.ioff()
        plt.show()


# ==========================================================
# 4) MAIN
# ==========================================================

def main():
    print(f"Loading dataset from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Gerekirse veri tipini string'e çevir, null kontrolü için
        if "Workout" in df.columns:
             df["Workout"] = df["Workout"].fillna("Null").astype(str)
        else:
             print("Error: 'Workout' column not found in CSV.")
             sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        sys.exit(1)

    print("Data loaded. Shape:", df.shape)

    # Özellikler
    print("Calculating features...")
    df = add_features(df)

    # Sınıflar (ORIGINAL GLOBAL QUANTILE CLASSIFICATION IS CALLED)
    print("Classifying intensity using global quantiles...")
    df = classify_intensity(df) # Calling the original classify_intensity function

    # Sonuç CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved classified dataset to: {OUTPUT_CSV}\n")

    # Sınıf istatistikleri
    print("=== Per-class counts (Global Quantile Intensity) ===")
    counts = df["intensity_class"].value_counts().sort_index()
    for c, cnt in counts.items():
        lbl = CLASS_LABELS.get(c, f"Class-{c}")
        print(f"{lbl:<35} -> {cnt:8d} samples")

    print("\n=== Per-class mean features (acc_mag, ENMO, intensity_score) ===")
    # Workout="Null" olmayanların ortalamasına bakalım ki Class 0 istatistikleri bozmasın
    df_ex = df[df["Workout"] != "Null"]
    
    for c in range(1, 6): # Sadece egzersiz sınıfları (1-5)
        sub = df_ex[df_ex["intensity_class"] == c]
        if len(sub) > 0:
            m_acc = sub["acc_mag"].mean()
            m_enmo = sub["ENMO"].mean()
            m_sc = sub["intensity_score"].mean()
            lbl = CLASS_LABELS.get(c)
            print(f"{lbl:<35} acc_mag={m_acc:.4f}  ENMO={m_enmo:.4f}  score={m_sc:.4f}")

    # Gerçek-zaman benzeri oynatma
    print("\nLaunching playback...")
    realtime_playback(df)


if __name__ == "__main__":
    main()