// GymBud_IMU_Node_RW.ino
// Common IMU + BLE peripheral code for Seeed XIAO nRF52840 Sense
// Change NODE_ID and DEVICE_NAME before uploading to each sensor board.

// ---------- CONFIG ----------
#define NODE_ID      3                  // 1 = left wrist, 2 = right wrist, 3 = chest
#define DEVICE_NAME  "GymBud_RW"        // will appear in BLE scanner

const float SAMPLE_RATE_HZ = 20.0f;     // IMU sample rate (20 Hz)
const unsigned long SAMPLE_INTERVAL_MS = (unsigned long)(1000.0f / SAMPLE_RATE_HZ);

// ---------- LIBRARIES ----------
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>      // IMU driver for XIAO nRF52840 Sense

// ---------- BLE UUIDs (custom, must MATCH central) ----------
const char* IMU_SERVICE_UUID = "7b8f0001-0000-4f79-a1af-59c5d5f0a001";
const char* IMU_CHAR_UUID    = "7b8f0002-0000-4f79-a1af-59c5d5f0a001";

// IMU payload: ax, ay, az, gx, gy, gz (floats) + nodeID (uint8)
struct ImuPacket {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
  uint8_t nodeId;
};

// BLE service + characteristic
BLEService imuService(IMU_SERVICE_UUID);
BLECharacteristic imuChar(
  IMU_CHAR_UUID,
  BLERead | BLENotify,
  sizeof(ImuPacket),
  true  // fixed length
);

unsigned long lastSampleMs = 0;

void setup() {
  // Serial for debugging
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for Serial
  }
  Serial.println("GymBud IMU Node starting...");

  // --------- IMU INIT ---------
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU (BMI270).");
    while (1) {
      delay(1000); // stop here if IMU fails
    }
  }
  Serial.println("IMU started.");

  // Optional: check which axes are available
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    Serial.println("Warning: IMU accel/gyro not fully available.");
  }

  // --------- BLE INIT ---------
  if (!BLE.begin()) {
    Serial.println("Failed to start BLE!");
    while (1) {
      delay(1000); // stop if BLE fails
    }
  }

  BLE.setLocalName(DEVICE_NAME);   // advertised device name
  BLE.setDeviceName(DEVICE_NAME);  // GAP device name
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(imuChar);
  BLE.addService(imuService);

  // initialize characteristic with zeros
  uint8_t empty[sizeof(ImuPacket)] = {0};
  imuChar.writeValue(empty, sizeof(empty));

  BLE.advertise();                 // start advertising
  Serial.print("BLE advertising as: ");
  Serial.println(DEVICE_NAME);
}

void loop() {
  // Keep BLE stack running
  BLE.poll();

  // Take a sample every SAMPLE_INTERVAL_MS
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return; // not yet time to sample
  }
  lastSampleMs = now;

  // --------- READ IMU ---------
  float ax, ay, az;
  float gx, gy, gz;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);   // g units
  } else {
    ax = ay = az = 0.0f;
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);      // dps
  } else {
    gx = gy = gz = 0.0f;
  }

  // --------- FILL PACKET ---------
  ImuPacket pkt;
  pkt.ax     = ax;
  pkt.ay     = ay;
  pkt.az     = az;
  pkt.gx     = gx;
  pkt.gy     = gy;
  pkt.gz     = gz;
  pkt.nodeId = (uint8_t)NODE_ID;

  // --------- SEND OVER BLE ---------
  imuChar.writeValue((uint8_t*)&pkt, sizeof(pkt));

  // Optional: debug print over USB
  Serial.print("ID:");
  Serial.print(pkt.nodeId);
  Serial.print(" ax:");
  Serial.print(pkt.ax, 3);
  Serial.print(" ay:");
  Serial.print(pkt.ay, 3);
  Serial.print(" az:");
  Serial.print(pkt.az, 3);
  Serial.print(" gx:");
  Serial.print(pkt.gx, 3);
  Serial.print(" gy:");
  Serial.print(pkt.gy, 3);
  Serial.print(" gz:");
  Serial.println(pkt.gz, 3);
}
