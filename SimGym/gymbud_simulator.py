import time
import csv
from collections import deque
import math
import random
import matplotlib.pyplot as plt

# ================== USER SETTINGS ==================

# Log dosyası adı
CSV_FILENAME = "gymbud_simulated_log.csv"

# Simülasyonda kullanacağımız node'lar ve eksenler
PLOT_NODE_IDS = [1, 2, 3]                       # 3 farklı IMU node'u
PLOT_AXES     = ["ax", "ay", "az", "gx", "gy", "gz"]  # 6 eksen
MAX_POINTS    = 300                             # Grafikte tutulacak son örnek sayısı

# Simülasyon örnekleme hızı (XIAO tarafında 20 Hz demiştik)
SIM_SAMPLE_RATE_HZ = 20.0
SIM_DT = 1.0 / SIM_SAMPLE_RATE_HZ

# ===================================================


def generate_fake_imu(node_id, t):
    """
    Belirli bir node ve zaman t için sahte ama 'mantıklı' IMU verisi üretir.
    - ax, ay: küçük sinüzoidal hareketler (+noise)
    - az: etrafında ~1 g (duran insan)
    - gx, gy, gz: yavaş dönme hareketleri
    """
    # Node'a bağlı faz farkı verelim ki grafikte farklı gözüksün
    phase = (node_id - 1) * math.pi / 4.0

    # Accel (g cinsinden)
    ax = 0.2 * math.sin(2 * math.pi * 0.5 * t + phase) + 0.02 * random.gauss(0, 1)
    ay = 0.2 * math.cos(2 * math.pi * 0.7 * t + phase) + 0.02 * random.gauss(0, 1)
    az = 1.0 + 0.05 * math.sin(2 * math.pi * 0.3 * t) + 0.02 * random.gauss(0, 1)

    # Gyro (deg/s cinsinden)
    gx = 10.0 * math.sin(2 * math.pi * 0.4 * t + phase) + 1.0 * random.gauss(0, 1)
    gy = 5.0  * math.cos(2 * math.pi * 0.6 * t + phase) + 1.0 * random.gauss(0, 1)
    gz = 3.0  * math.sin(2 * math.pi * 0.2 * t + phase) + 1.0 * random.gauss(0, 1)

    return ax, ay, az, gx, gy, gz


def main():
    print("Starting GymBud IMU SIMULATOR (no serial needed).")
    print("Logging to:", CSV_FILENAME)

    # CSV hazırlığı
    csv_file = open(CSV_FILENAME, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["node_id", "timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]
    csv_writer.writerow(header)
    csv_file.flush()

    # ---------- Real-time plot yapısı ----------
    # Her node için x (sample index)
    x_data = {
        node_id: deque(maxlen=MAX_POINTS)
        for node_id in PLOT_NODE_IDS
    }

    # Her (node, axis) çifti için y değerleri
    y_data = {
        (node_id, axis): deque(maxlen=MAX_POINTS)
        for node_id in PLOT_NODE_IDS
        for axis in PLOT_AXES
    }

    # Her node için ayrı sample sayacı
    sample_idx = {node_id: 0 for node_id in PLOT_NODE_IDS}

    plt.ion()  # interactive mode

    # 6 eksen için 3x2 subplot
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs = axs.flatten()

    # Her eksen + her node için Line objesi
    lines = {}  # key: (axis_name, node_id) -> Line2D

    for i, axis_name in enumerate(PLOT_AXES):
        ax_plot = axs[i]
        for node_id in PLOT_NODE_IDS:
            line, = ax_plot.plot([], [], label=f"node {node_id}")
            lines[(axis_name, node_id)] = line

        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel(axis_name)
        ax_plot.set_title(f"{axis_name} – SIM IMU")
        ax_plot.grid(True)
        ax_plot.legend()

    fig.suptitle("GymBud Real-time IMU SIMULATION (3 nodes, 6 axes)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Simülasyon zamanı
    t = 0.0
    sim_start = time.time()

    try:
        print("Simulation running. Close the plot window or Ctrl+C to stop.")
        while True:
            loop_start = time.time()

            for node_id in PLOT_NODE_IDS:
                # Sahte IMU verisi üret
                ax_val, ay_val, az_val, gx_val, gy_val, gz_val = generate_fake_imu(node_id, t)
                t_ms = int((time.time() - sim_start) * 1000)

                # CSV'ye yaz (gerçek sistemde Arduino'dan gelen satırla birebir aynı format)
                csv_writer.writerow([node_id, t_ms, ax_val, ay_val, az_val,
                                     gx_val, gy_val, gz_val])
                csv_file.flush()

                # Plot datasını güncelle
                idx = sample_idx[node_id]
                sample_idx[node_id] += 1
                x_data[node_id].append(idx)

                axis_values = {
                    "ax": ax_val,
                    "ay": ay_val,
                    "az": az_val,
                    "gx": gx_val,
                    "gy": gy_val,
                    "gz": gz_val,
                }

                for axis_name in PLOT_AXES:
                    y_data[(node_id, axis_name)].append(axis_values[axis_name])

            # ---- Çizgileri güncelle ----
            for i, axis_name in enumerate(PLOT_AXES):
                ax_plot = axs[i]
                all_y_for_axis = []
                any_x = []

                for nid in PLOT_NODE_IDS:
                    line = lines[(axis_name, nid)]
                    xs = list(x_data[nid])
                    ys = list(y_data[(nid, axis_name)])
                    line.set_xdata(xs)
                    line.set_ydata(ys)
                    all_y_for_axis.extend(ys)
                    any_x.extend(xs)

                if any_x and all_y_for_axis:
                    ax_plot.set_xlim(min(any_x), max(any_x))
                    ymin = min(all_y_for_axis)
                    ymax = max(all_y_for_axis)
                    if ymin == ymax:
                        ymin -= 1.0
                        ymax += 1.0
                    ax_plot.set_ylim(
                        ymin - 0.1 * abs(ymin if ymin != 0 else 1),
                        ymax + 0.1 * abs(ymax if ymax != 0 else 1),
                    )

            plt.draw()
            plt.pause(0.001)

            # 20 Hz civarı hızda çalışması için uyku
            loop_duration = time.time() - loop_start
            sleep_time = SIM_DT - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

            t += SIM_DT

    except KeyboardInterrupt:
        print("\nSimulation stopped by user (KeyboardInterrupt).")

    finally:
        csv_file.close()
        print("CSV file saved:", CSV_FILENAME)


if __name__ == "__main__":
    main()
