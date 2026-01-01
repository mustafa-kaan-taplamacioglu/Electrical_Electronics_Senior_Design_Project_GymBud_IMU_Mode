// GymBud_Central_Hub.ino
// Multi-connection BLE Central hub for GymBud (LW + RW)
// Also advertises a name (GymBud_Central) so it is visible to LightBlue, etc.

#include <ArduinoBLE.h>

// ---- Target peripheral names ----
const char* TARGET_NAMES[2] = { "GymBud_LW", "GymBud_RW" };

// ---- UUIDs (must match peripherals) ----
const char* IMU_SERVICE_UUID = "7b8f0001-0000-4f79-a1af-59c5d5f0a001";
const char* IMU_CHAR_UUID    = "7b8f0002-0000-4f79-a1af-59c5d5f0a001";

// Dummy advertised service so Central is also BLE-discoverable by name
const char* HUB_AD_SERVICE_UUID = "7b8f1000-0000-4f79-a1af-59c5d5f0a001";
BLEService hubAdvertisedService(HUB_AD_SERVICE_UUID);

// Must match peripheral struct layout (byte-for-byte)
struct __attribute__((packed)) ImuPacket {
  float ax, ay, az;
  float gx, gy, gz;
  uint8_t nodeId;
};

struct Link {
  const char* name;
  BLEDevice dev;
  BLECharacteristic imuChar;
  bool connected;
  bool subscribed;

  // Latest received data cache
  ImuPacket lastPkt;
  bool havePkt;

  // Print scheduler
  unsigned long nextPrintMs;
};

Link links[2] = {
  { "GymBud_LW", BLEDevice(), BLECharacteristic(), false, false, {}, false, 0 },
  { "GymBud_RW", BLEDevice(), BLECharacteristic(), false, false, {}, false, 0 }
};

bool scanning = false;

static inline void printPacket(const ImuPacket& pkt) {
  // EXACT same output format as before
  Serial.print("ID:");
  Serial.print(pkt.nodeId);
  Serial.print(" ax:");
  Serial.print(pkt.ax, 4);
  Serial.print(" ay:");
  Serial.print(pkt.ay, 4);
  Serial.print(" az:");
  Serial.print(pkt.az, 4);
  Serial.print(" gx:");
  Serial.print(pkt.gx, 4);
  Serial.print(" gy:");
  Serial.print(pkt.gy, 4);
  Serial.print(" gz:");
  Serial.println(pkt.gz, 4);
}

void startAdvertisingAsHub() {
  BLE.setLocalName("GymBud_Central");
  BLE.setDeviceName("GymBud_Central");

  BLE.setAdvertisedService(hubAdvertisedService);
  BLE.addService(hubAdvertisedService);

  BLE.advertise();
  Serial.println(">> Advertising as: GymBud_Central");
}

void startScan() {
  if (scanning) return;
  BLE.scan();
  scanning = true;
  Serial.println(">> Scanning for GymBud_LW / GymBud_RW ...");
}

void stopScan() {
  if (!scanning) return;
  BLE.stopScan();
  scanning = false;
  Serial.println(">> Scan stopped");
}

int findLinkIndexByName(const String& name) {
  for (int i = 0; i < 2; i++) {
    if (name == links[i].name) return i;
  }
  return -1;
}

bool allConnected() {
  for (int i = 0; i < 2; i++) {
    if (!links[i].connected || !links[i].subscribed) return false;
  }
  return true;
}

bool connectAndSubscribe(int idx, BLEDevice peripheral) {
  Serial.print("Connecting to ");
  Serial.print(links[idx].name);
  Serial.print(" @ ");
  Serial.println(peripheral.address());

  stopScan();

  if (!peripheral.connect()) {
    Serial.println("  ERROR: connect failed");
    startScan();
    return false;
  }

  if (!peripheral.discoverAttributes()) {
    Serial.println("  ERROR: discoverAttributes failed");
    peripheral.disconnect();
    startScan();
    return false;
  }

  BLECharacteristic c = peripheral.characteristic(IMU_CHAR_UUID);
  if (!c) {
    Serial.println("  ERROR: IMU characteristic not found");
    peripheral.disconnect();
    startScan();
    return false;
  }

  if (!c.canSubscribe()) {
    Serial.println("  ERROR: IMU characteristic not notifiable");
    peripheral.disconnect();
    startScan();
    return false;
  }

  if (!c.subscribe()) {
    Serial.println("  ERROR: subscribe failed");
    peripheral.disconnect();
    startScan();
    return false;
  }

  links[idx].dev = peripheral;
  links[idx].imuChar = c;
  links[idx].connected = true;
  links[idx].subscribed = true;

  links[idx].havePkt = false;

  // Start printing at a clean boundary
  unsigned long now = millis();
  links[idx].nextPrintMs = now;

  Serial.print("  OK: Subscribed to ");
  Serial.println(links[idx].name);

  if (!allConnected()) startScan();
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1200);

  Serial.println("GymBud Central Hub starting...");

  if (!BLE.begin()) {
    Serial.println("ERROR: BLE.begin() failed");
    while (1) delay(1000);
  }

  startAdvertisingAsHub();
  startScan();
}

void loop() {
  BLE.poll();

  const unsigned long PRINT_PERIOD_MS = 50; // 20 Hz

  // 1) Keep connections healthy; if any drops, mark and rescan
  for (int i = 0; i < 2; i++) {
    if (links[i].connected && !links[i].dev.connected()) {
      Serial.print("!! Disconnected: ");
      Serial.println(links[i].name);
      links[i].connected = false;
      links[i].subscribed = false;
      links[i].havePkt = false;
      startScan();
    }
  }

  // 2) While scanning, try to connect to any missing target
  if (scanning) {
    BLEDevice p = BLE.available();
    if (p) {
      String name = p.localName();
      int idx = findLinkIndexByName(name);
      if (idx >= 0 && !links[idx].connected) {
        connectAndSubscribe(idx, p);
      }
    }
  }

  unsigned long now = millis();

  // 3) Drain notifications ASAP into cache (fast, avoids losing updates)
  for (int i = 0; i < 2; i++) {
    if (!links[i].connected || !links[i].subscribed) continue;

    BLECharacteristic &c = links[i].imuChar;
    if (c.valueUpdated() && c.valueLength() == sizeof(ImuPacket)) {
      c.readValue((uint8_t*)&links[i].lastPkt, sizeof(ImuPacket));
      links[i].havePkt = true;
    }
  }

  // 4) Print at exactly 20 Hz per link using the latest cached packet
  for (int i = 0; i < 2; i++) {
    if (!links[i].connected || !links[i].subscribed) continue;
    if (!links[i].havePkt) continue;

    // drift-free print scheduler
    if ((long)(now - links[i].nextPrintMs) < 0) continue;

    // catch up without drift
    do {
      links[i].nextPrintMs += PRINT_PERIOD_MS;
    } while ((long)(now - links[i].nextPrintMs) >= 0);

    printPacket(links[i].lastPkt); // SAME output format
  }
}
