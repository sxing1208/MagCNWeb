/*
  ESP32-S3 + 4x CD74HC4067 — MagCNWeb-compatible 64-channel scanner

  Operation:
    1. ESP32-S3 creates a Wi-Fi access point.
    2. The host computer connects to that access point.
    3. MagCNWeb runs on the host computer:
         py -V:3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
    4. ESP32 connects to:
         ws://192.168.4.2:8000/ws
    5. Each frame is transmitted as:
         {
           "pid": "device",
           "key": "admin",
           "value": "v0,v1,...,v63"
         }

  MagCNWeb reshapes the 64 values into an 8x8 matrix using row-major order.

  ADC connections:
    MUX1 COM -> IO1
    MUX2 COM -> IO2
    MUX3 COM -> IO12
    MUX4 COM -> IO4

  Select lines:
    MUX1 S0..S3 -> IO5, IO6, IO7, IO8
    MUX2 S0..S3 -> IO9, IO10, IO11, IO47
    MUX3 S0..S3 -> IO13, IO14, IO21, IO35
    MUX4 S0..S3 -> IO36, IO37, IO38, IO48

  Required Arduino libraries:
    - WiFi                  (included with ESP32 Arduino core)
    - WebSockets by Markus Sattler
      Library Manager name: "WebSockets"
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>

// -----------------------------------------------------------------------------
// Wi-Fi access point configuration
// -----------------------------------------------------------------------------

static const char *AP_SSID     = "MagCN";
static const char *AP_PASSWORD = "magcnadmin";

// Typical first computer connected to an ESP32 AP.
// Change this if ipconfig shows a different host IPv4 address.
static const char *MAGCN_SERVER_IP = "192.168.4.2";

static const uint16_t MAGCN_SERVER_PORT = 8000;
static const char *MAGCN_WS_PATH = "/ws";

// Give the ESP32 AP a predictable address.
IPAddress AP_IP(192, 168, 4, 1);
IPAddress AP_GATEWAY(192, 168, 4, 1);
IPAddress AP_SUBNET(255, 255, 255, 0);

WebSocketsClient webSocket;

// -----------------------------------------------------------------------------
// Multiplexer configuration
// -----------------------------------------------------------------------------

static constexpr uint8_t NUM_MUX = 4;
static constexpr uint8_t CHANNELS_PER_MUX = 16;
static constexpr uint8_t NUM_CHANNELS =
    NUM_MUX * CHANNELS_PER_MUX;

// MUX COM/ADC pins
static const uint8_t MUX_ADC_PIN[NUM_MUX] = {
  1,   // MUX1 COM
  2,   // MUX2 COM
  12,  // MUX3 COM
  4    // MUX4 COM
};

// S0..S3, ordered LSB to MSB
static const uint8_t MUX_SEL_PINS[NUM_MUX][4] = {
  {  5,  6,  7,  8 },   // MUX1
  {  9, 10, 11, 47 },   // MUX2
  { 13, 14, 21, 35 },   // MUX3
  { 36, 37, 38, 48 }    // MUX4
};

// -----------------------------------------------------------------------------
// ADC and timing configuration
// -----------------------------------------------------------------------------

static constexpr uint8_t ADC_BITS = 12;
static constexpr adc_attenuation_t ADC_ATTEN = ADC_11db;

// One settling delay is used for all four multiplexers because their select
// lines are updated together.
static constexpr uint32_t SETTLE_US = 5000;

static constexpr uint8_t OVERSAMPLE = 1;

// MagCNWeb workers_sync.py and dummy.py use 5 samples/second.
static constexpr uint32_t SAMPLE_RATE_HZ = 5;
static constexpr uint32_t FRAME_PERIOD_MS =
    1000UL / SAMPLE_RATE_HZ;

static uint32_t nextFrameTime = 0;

// Raw 12-bit ADC frame
static uint16_t frame[NUM_CHANNELS];

// Large enough for JSON plus 64 values of up to four digits each.
static char payload[512];

// -----------------------------------------------------------------------------
// Multiplexer and ADC functions
// -----------------------------------------------------------------------------

void setAllMuxChannels(uint8_t channel) {
  for (uint8_t mux = 0; mux < NUM_MUX; ++mux) {
    digitalWrite(
      MUX_SEL_PINS[mux][0],
      (channel >> 0) & 0x01
    );
    digitalWrite(
      MUX_SEL_PINS[mux][1],
      (channel >> 1) & 0x01
    );
    digitalWrite(
      MUX_SEL_PINS[mux][2],
      (channel >> 2) & 0x01
    );
    digitalWrite(
      MUX_SEL_PINS[mux][3],
      (channel >> 3) & 0x01
    );
  }
}

uint16_t readADC(uint8_t pin) {
  uint32_t accumulator = 0;

  for (uint8_t sample = 0; sample < OVERSAMPLE; ++sample) {
    accumulator += analogRead(pin);
  }

  return static_cast<uint16_t>(accumulator / OVERSAMPLE);
}

void acquireFrame() {
  /*
    All four multiplexers are switched to the same channel simultaneously.

    Storage remains mux-major:
      frame[0..15]  = MUX1 channels 0..15
      frame[16..31] = MUX2 channels 0..15
      frame[32..47] = MUX3 channels 0..15
      frame[48..63] = MUX4 channels 0..15
  */

  for (uint8_t channel = 0;
       channel < CHANNELS_PER_MUX;
       ++channel) {

    setAllMuxChannels(channel);
    delayMicroseconds(SETTLE_US);

    for (uint8_t mux = 0; mux < NUM_MUX; ++mux) {
      const uint8_t index =
          mux * CHANNELS_PER_MUX + channel;

      frame[index] = readADC(MUX_ADC_PIN[mux]);
    }

    // Service WebSocket control frames during acquisition.
    webSocket.loop();
  }
}

// -----------------------------------------------------------------------------
// JSON creation and transmission
// -----------------------------------------------------------------------------

bool buildPayload() {
  int used = snprintf(
    payload,
    sizeof(payload),
    "{\"pid\":\"device\",\"key\":\"admin\",\"value\":\""
  );

  if (used < 0 || used >= static_cast<int>(sizeof(payload))) {
    return false;
  }

  for (uint8_t i = 0; i < NUM_CHANNELS; ++i) {
    const int written = snprintf(
      payload + used,
      sizeof(payload) - used,
      (i == 0) ? "%u" : ",%u",
      static_cast<unsigned int>(frame[i])
    );

    if (written < 0 ||
        written >= static_cast<int>(sizeof(payload) - used)) {
      return false;
    }

    used += written;
  }

  const int written = snprintf(
    payload + used,
    sizeof(payload) - used,
    "\"}"
  );

  return written >= 0 &&
         written < static_cast<int>(sizeof(payload) - used);
}

void sendFrame() {
  if (!webSocket.isConnected()) {
    return;
  }

  if (!buildPayload()) {
    Serial.println(F("[ERROR] JSON payload buffer too small"));
    return;
  }

  webSocket.sendTXT(payload);

  // Optional local debugging
  Serial.println(payload);
}

// -----------------------------------------------------------------------------
// WebSocket events
// -----------------------------------------------------------------------------

void webSocketEvent(
    WStype_t type,
    uint8_t *incomingPayload,
    size_t length) {

  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println(F("[WS] Disconnected"));
      break;

    case WStype_CONNECTED:
      Serial.printf(
        "[WS] Connected to ws://%s:%u%s\n",
        MAGCN_SERVER_IP,
        MAGCN_SERVER_PORT,
        MAGCN_WS_PATH
      );
      break;

    case WStype_TEXT:
      Serial.printf(
        "[WS] Received: %.*s\n",
        static_cast<int>(length),
        reinterpret_cast<char *>(incomingPayload)
      );
      break;

    case WStype_ERROR:
      Serial.println(F("[WS] Error"));
      break;

    case WStype_PING:
      Serial.println(F("[WS] Ping received"));
      break;

    case WStype_PONG:
      Serial.println(F("[WS] Pong received"));
      break;

    default:
      break;
  }
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

void configureMuxes() {
  for (uint8_t mux = 0; mux < NUM_MUX; ++mux) {
    for (uint8_t select = 0; select < 4; ++select) {
      pinMode(MUX_SEL_PINS[mux][select], OUTPUT);
      digitalWrite(MUX_SEL_PINS[mux][select], LOW);
    }
  }
}

void configureADC() {
  analogReadResolution(ADC_BITS);

  for (uint8_t mux = 0; mux < NUM_MUX; ++mux) {
    pinMode(MUX_ADC_PIN[mux], INPUT);
    analogSetPinAttenuation(
      MUX_ADC_PIN[mux],
      ADC_ATTEN
    );
  }
}

void startAccessPoint() {
  WiFi.mode(WIFI_AP);

  if (!WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_SUBNET)) {
    Serial.println(F("[WiFi] AP address configuration failed"));
  }

  if (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println(F("[WiFi] Failed to create access point"));
    return;
  }

  Serial.println();
  Serial.println(F("[WiFi] Access point started"));
  Serial.printf("[WiFi] SSID: %s\n", AP_SSID);
  Serial.printf("[WiFi] ESP32 IP: %s\n",
                WiFi.softAPIP().toString().c_str());
  Serial.printf("[WiFi] Expected host IP: %s\n",
                MAGCN_SERVER_IP);
}

void startWebSocket() {
  webSocket.begin(
    MAGCN_SERVER_IP,
    MAGCN_SERVER_PORT,
    MAGCN_WS_PATH
  );

  webSocket.onEvent(webSocketEvent);

  // Retry if MagCNWeb starts after the ESP32.
  webSocket.setReconnectInterval(5000);

  // Maintain the connection and detect an unavailable host.
  webSocket.enableHeartbeat(
    15000,  // send ping every 15 seconds
    3000,   // wait 3 seconds for pong
    2       // disconnect after two missed pongs
  );
}

// -----------------------------------------------------------------------------
// Arduino setup and loop
// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println(F("# MagCNWeb 64-channel peripheral starting"));

  configureMuxes();
  configureADC();
  startAccessPoint();
  startWebSocket();

  nextFrameTime = millis();
}

void loop() {
  webSocket.loop();

  const uint32_t now = millis();

  if (static_cast<int32_t>(now - nextFrameTime) >= 0) {
    // Advance relative to the intended schedule to reduce timing drift.
    nextFrameTime += FRAME_PERIOD_MS;

    acquireFrame();
    sendFrame();

    // Recover if acquisition/debugging caused us to fall far behind.
    if (static_cast<int32_t>(millis() - nextFrameTime) >=
        static_cast<int32_t>(FRAME_PERIOD_MS)) {
      nextFrameTime = millis() + FRAME_PERIOD_MS;
    }
  }

  delay(1);
}
