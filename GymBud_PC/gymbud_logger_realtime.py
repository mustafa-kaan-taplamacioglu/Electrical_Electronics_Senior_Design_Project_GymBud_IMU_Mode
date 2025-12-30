import serial
import time
import csv
from collections import deque
import matplotlib.pyplot as plt

# PORT = "/dev/tty.usbmodem1101"   # ← kendi port ismini yaz
# CSV_FILENAME = "gymbud_log.csv"  # ← istersen değiştir
# PLOT_NODE_ID = 1                 # ← 1,2,3: hangi node’u izlemek istiyorsun
# PLOT_AXIS    = "ax"              # ← hangi eksen

# ================== USER SETTINGS ==================

# !!! BURAYI KENDİ PORTUNA GÖRE DEĞİŞTİR !!!
# Windows örnek: "COM5"
# Mac örnek: "/dev/tty.usbmodem1101"
PORT = "/dev/tty.usbmodem1101"

BAUDRATE = 115200

# Log dosyası adı – istersen değiştir
CSV_FILENAME = "gymbud_log.csv"

# Real-time plot için:
PLOT_NODE_IDS = [1, 2, 3]                       # Aynı anda göreceğin node'lar
PLOT_AXES     = ["ax", "ay", "az", "gx", "gy", "gz"]  # Aynı anda göreceğin eksenler
MAX_POINTS    = 300                             # Grafikte tutulacak son örnek sayısı

# ===================================================


def main():
    print("Opening serial port:", PORT)
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)

    # Arduino reset'lenip header basması için kısa bekleme
    time.sleep(2.0)

    # Serial buffer temizle
    ser.reset_input_buffer()

    print("Logging to:", CSV_FILENAME)
    csv_file = open(CSV_FILENAME, mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    # CSV header (Arduino ile uyumlu)
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

    # Her eksen için, her node'a ait line objesini tutacağız
    lines = {}  # key: (axis_name, node_id) -> Line2D

    for i, axis_name in enumerate(PLOT_AXES):
        ax_plot = axs[i]
        for node_id in PLOT_NODE_IDS:
            line, = ax_plot.plot([], [], label=f"node {node_id}")
            lines[(axis_name, node_id)] = line

        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel(axis_name)
        ax_plot.set_title(f"{axis_name} – IMU")
        ax_plot.grid(True)
        ax_plot.legend()

    fig.suptitle("GymBud Real-time IMU (3 nodes, 6 axes)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        print("Starting read loop. Ctrl+C to stop.")
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                # timeout – veri yoksa döngüye devam
                continue

            try:
                line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            except UnicodeDecodeError:
                continue

            if not line_str:
                continue

            # Arduino'dan gelen header satırını (node_id,...) atla
            if line_str.startswith("node_id"):
                print("Serial header from Arduino:", line_str)
                continue

            # Beklenen format:
            # node_id,timestamp_ms,ax,ay,az,gx,gy,gz
            parts = line_str.split(",")
            if len(parts) != 8:
                # Hatalı satır – debug için istersen print edebilirsin:
                # print("Malformed line:", line_str)
                continue

            try:
                node_id = int(parts[0])
                t_ms    = int(parts[1])
                ax_val  = float(parts[2])
                ay_val  = float(parts[3])
                az_val  = float(parts[4])
                gx_val  = float(parts[5])
                gy_val  = float(parts[6])
                gz_val  = float(parts[7])
            except ValueError:
                # Tip dönüşümü başarısızsa satırı atla
                # print("Parse error on line:", line_str)
                continue

            # CSV'ye yaz
            csv_writer.writerow([node_id, t_ms, ax_val, ay_val, az_val,
                                 gx_val, gy_val, gz_val])
            csv_file.flush()

            # ---- Plot datasını güncelle (sadece seçilen node'larsa) ----
            if node_id in PLOT_NODE_IDS:
                # Node'un sample index'ini güncelle
                idx = sample_idx[node_id]
                sample_idx[node_id] += 1

                x_data[node_id].append(idx)

                # Her eksen için ilgili deque'e ekle
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

                    # Bu eksen için tüm node'ların verilerini çiz
                    all_y_for_axis = []

                    for nid in PLOT_NODE_IDS:
                        line = lines[(axis_name, nid)]
                        xs = list(x_data[nid])
                        ys = list(y_data[(nid, axis_name)])
                        line.set_xdata(xs)
                        line.set_ydata(ys)
                        all_y_for_axis.extend(ys)

                    # x/y limitlerini dinamik ayarla
                    # (en az bir node veri göndermişse)
                    any_x = []
                    for nid in PLOT_NODE_IDS:
                        any_x.extend(x_data[nid])

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
                plt.pause(0.001)  # küçük bir delay, GUI için

    except KeyboardInterrupt:
        print("\nStopping logging (KeyboardInterrupt).")

    finally:
        ser.close()
        csv_file.close()
        print("Serial port closed. CSV file saved:", CSV_FILENAME)


if __name__ == "__main__":
    main()
