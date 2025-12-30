// GymBud_Central_Hub.ino
// BLE Central hub for GymBud project
// Connects to GymBud_IMU_Node peripherals and forwards IMU data over Serial.

#include <ArduinoBLE.h>

// Same UUIDs as peripheral
const char* IMU_SERVICE_UUID = "7b8f0001-0000-4f79-a1af-59c5d5f0a001";
const char* IMU_CHAR_UUID    = "7b8f0002-0000-4f79-a1af-59c5d5f0a001";

// Must match ImuPacket in peripheral
struct ImuPacket {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
  uint8_t nodeId;
};

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for Serial
  }
  Serial.println("GymBud Central Hub starting...");

  if (!BLE.begin()) {
    Serial.println("Failed to start BLE.");
    while (1) {
      delay(1000);
    }
  }

  // Central mode only
  BLE.setLocalName("GymBud_Central");
  Serial.println("BLE Central started. Scanning for GymBud nodes...");

  // CSV header for Python logger
  Serial.println("node_id,timestamp_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  // Non-blocking scan for peripheral
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    // Optional filter: only devices with our service
    if (!peripheral.hasServiceUuid(IMU_SERVICE_UUID)) {
      return; // not a GymBud node
    }

    Serial.print("Found node: ");
    Serial.print(peripheral.address());
    Serial.print(" (");
    Serial.print(peripheral.localName());
    Serial.println(")");

    // Connect
    if (!peripheral.connect()) {
      Serial.println("Failed to connect.");
      return;
    }
    Serial.println("Connected.");

    // Discover service & characteristic
    if (!peripheral.discoverService(IMU_SERVICE_UUID)) {
      Serial.println("IMU service not found. Disconnecting.");
      peripheral.disconnect();
      return;
    }

    BLECharacteristic imuChar = peripheral.characteristic(IMU_CHAR_UUID);
    if (!imuChar) {
      Serial.println("IMU characteristic not found. Disconnecting.");
      peripheral.disconnect();
      return;
    }

    if (!imuChar.canSubscribe()) {
      Serial.println("IMU characteristic is not notifiable. Disconnecting.");
      peripheral.disconnect();
      return;
    }

    imuChar.subscribe(); // enable notifications

    // Read loop while connected
    while (peripheral.connected()) {
      BLE.poll(); // keep BLE stack alive

      if (imuChar.valueUpdated()) {
        ImuPacket pkt;
        if (imuChar.valueLength() == sizeof(ImuPacket)) {
          imuChar.readValue((uint8_t*)&pkt, sizeof(pkt));

          unsigned long t = millis();

          // Print one CSV line
          Serial.print(pkt.nodeId);
          Serial.print(",");
          Serial.print(t);
          Serial.print(",");
          Serial.print(pkt.ax, 5);
          Serial.print(",");
          Serial.print(pkt.ay, 5);
          Serial.print(",");
          Serial.print(pkt.az, 5);
          Serial.print(",");
          Serial.print(pkt.gx, 5);
          Serial.print(",");
          Serial.print(pkt.gy, 5);
          Serial.print(",");
          Serial.println(pkt.gz, 5);
        }
      }
    }

    Serial.println("Peripheral disconnected. Scanning again...");
  }

  // Keep BLE stack alive even when idle
  BLE.poll();
}
