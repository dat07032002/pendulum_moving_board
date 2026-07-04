/*
 * sensor_test.ino — AS5600 magnet diagnostic plus BNO086 presence.
 * ESP32: AS5600 Wire SDA=21/SCL=22; BNO086 Wire1 SDA=32/SCL=33; both 400 kHz.
 *
 * Streams the AS5600 STATUS + raw angle at 10 Hz so you can reposition the
 * pendulum magnet and watch it lock on:
 *   MD=1 -> magnet detected (good). MD=0 -> not seen.
 *   ML=1 -> magnet too weak/far.    MH=1 -> too strong/close.
 * When healthy: MD=1, ML=0, MH=0, and `raw` sweeps 0..4095 as you rotate the pole.
 */
#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define BNO_SDA_PIN 32
#define BNO_SCL_PIN 33
#define AS5600_ADDR 0x36
#define BNO_ADDR_A  0x4A   // BNO086 default
#define BNO_ADDR_B  0x4B   // BNO086 alternate

uint16_t as5600_read16(uint8_t reg) {          // 12-bit value at reg:reg+1
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFFFF;
  Wire.requestFrom(AS5600_ADDR, (uint8_t)2);
  if (Wire.available() < 2) return 0xFFFF;
  uint16_t v = ((uint16_t)Wire.read() << 8) | Wire.read();
  return v & 0x0FFF;
}
uint8_t as5600_read8(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  Wire.requestFrom(AS5600_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}
bool i2c_present(TwoWire &bus, uint8_t addr) {
  bus.beginTransmission(addr);
  return bus.endTransmission() == 0;
}

void setup() {
  Serial.begin(921600);
  delay(400);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  Wire1.begin(BNO_SDA_PIN, BNO_SCL_PIN);
  Wire1.setClock(400000);
  Serial.println();
  Serial.println("# SENSOR TEST  (AS5600 Wire=21/22, BNO086 Wire1=32/33, 400kHz)");
  Serial.print("# Wire scan:");
  for (uint8_t a = 1; a < 127; a++)
    if (i2c_present(Wire, a)) { Serial.print(" 0x"); Serial.print(a, HEX); }
  Serial.println();
  Serial.print("# Wire1 scan:");
  for (uint8_t a = 1; a < 127; a++)
    if (i2c_present(Wire1, a)) { Serial.print(" 0x"); Serial.print(a, HEX); }
  Serial.println();
  Serial.print("#   AS5600@0x36: "); Serial.println(i2c_present(Wire, AS5600_ADDR) ? "FOUND" : "MISSING");
  Serial.print("#   BNO086@0x4A/0x4B: ");
  Serial.println(i2c_present(Wire1, BNO_ADDR_A) ? "FOUND@0x4A" : (i2c_present(Wire1, BNO_ADDR_B) ? "FOUND@0x4B" : "MISSING"));
  Serial.println("# streaming AS5600 @10Hz -> reposition magnet until MD=1, ML=0, MH=0, and raw sweeps as you rotate");
}

void loop() {
  uint8_t st  = as5600_read8(0x0B);
  uint8_t agc = as5600_read8(0x1A);
  uint16_t mg = as5600_read16(0x1B);
  uint16_t rw = as5600_read16(0x0C);
  Serial.print("raw="); Serial.print(rw);
  Serial.print("  deg="); Serial.print(rw * 360.0f / 4096.0f, 1);
  Serial.print("  MD="); Serial.print((st >> 5) & 1);
  Serial.print(" ML="); Serial.print((st >> 4) & 1);
  Serial.print(" MH="); Serial.print((st >> 3) & 1);
  Serial.print("  AGC="); Serial.print(agc);
  Serial.print("  MAGNITUDE="); Serial.println(mg);
  delay(100);
}
