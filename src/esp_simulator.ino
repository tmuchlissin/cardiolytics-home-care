#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>

const char* ssid = "Default User";
const char* password = "0192837465";
const char* mqtt_server = "192.168.0.111";
const int mqtt_port = 1883;
const char* mqtt_user = "espuser";
const char* mqtt_password = "esp32mqtt!";
const char* device_id = "BP0001"; // This can be changed to any BPXXXX value
String mqtt_topic = "bp_monitor/" + String(device_id);
String command_topic = "bp_monitor/command/" + String(device_id);
String status_topic = "bp_monitor/status/" + String(device_id);

WiFiClient espClient;
PubSubClient client(espClient);
int systolic = 0, diastolic = 0, pulse = 0;
bool isConnected = false;
bool previousConnectionStatus = false;

const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 7 * 3600; // WIB is UTC+7
const int daylightOffset_sec = 0;

void setup_wifi() {
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected, IP: " + WiFi.localIP().toString());
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    isConnected = true;
  } else {
    Serial.println("\nFailed to connect to WiFi");
    isConnected = false;
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (int i = 0; i < length; i++) message += (char)payload[i];
  Serial.println("Message received on topic [" + String(topic) + "]: " + message);

  DynamicJsonDocument doc(256);
  DeserializationError error = deserializeJson(doc, message);
  if (error) {
    Serial.println("Failed to parse JSON: " + String(error.c_str()));
    return;
  }

  if (doc.containsKey("command") && doc["command"] == "start_measurement") {
    Serial.println("Received start_measurement command");
    simulate_data();
    send_data();
  } else {
    Serial.println("Unknown command or malformed message");
  }
}

void reconnect() {
  while (!client.connected() && isConnected) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP32Client-" + String(random(0xffff), HEX);
    client.setKeepAlive(5);
    // Construct dynamic LWT payload using device_id
    String lwtPayload = "{\"device_id\":\"" + String(device_id) + "\",\"status\":\"disconnected\"}";
    if (client.connect(
      clientId.c_str(),
      mqtt_user,
      mqtt_password,
      status_topic.c_str(), // topic LWT
      0,                    // QoS
      false,                // retained
      lwtPayload.c_str()    // dynamic LWT payload
    )) {
      Serial.println("connected");
      client.subscribe(command_topic.c_str());
      Serial.println("Subscribed to " + command_topic);
      send_connection_status(true);
    } else {
      Serial.println("failed, rc=" + String(client.state()) + " retrying in 5s");
      delay(5000);
    }
  }
}

String getCurrentTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return "2025-06-07T08:54:00Z";
  }
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
  return String(buffer);
}

void simulate_data() {
  int bp_category = random(0, 5);  // 0 = Hypotension, 1 = Normal, 2 = Prehypertension, 3 = Stage 1 Hypertension, 4 = Stage 2 Hypertension
  int hr_category = random(0, 3);  // 0 = Bradycardia, 1 = Normal, 2 = Tachycardia

  switch (bp_category) {
    case 0: // Hypotension
      systolic = random(70, 90);
      diastolic = random(40, 60);
      break;
    case 1: // Normal
      systolic = random(90, 120);
      diastolic = random(60, 80);
      break;
    case 2: // Prehypertension
      systolic = random(120, 140);
      diastolic = random(80, 90);
      break;
    case 3: // Stage 1 Hypertension
      systolic = random(140, 160);
      diastolic = random(90, 100);
      break;
    case 4: // Stage 2 Hypertension
      systolic = random(160, 181);
      diastolic = random(100, 111);
      break;
  }

  switch (hr_category) {
    case 0: // Bradycardia
      pulse = random(40, 60);
      break;
    case 1: // Normal
      pulse = random(60, 101);
      break;
    case 2: // Tachycardia
      pulse = random(101, 140);
      break;
  }

  Serial.println("Simulated data: systolic=" + String(systolic) + ", diastolic=" + String(diastolic) + ", pulse=" + String(pulse));
}

void send_data() {
  if (!client.connected()) {
    Serial.println("Not connected to MQTT, skipping send_data");
    return;
  }
  DynamicJsonDocument doc(256);
  doc["device_id"] = device_id;
  doc["systolic"] = systolic;
  doc["diastolic"] = diastolic;
  doc["pulse"] = pulse;
  doc["timestamp"] = getCurrentTimestamp();
  char buffer[256];
  serializeJson(doc, buffer);
  if (client.publish(mqtt_topic.c_str(), buffer)) {
    Serial.println("Data sent to " + mqtt_topic + ": " + String(buffer));
  } else {
    Serial.println("Failed to send data to " + mqtt_topic);
  }
}

void send_connection_status(bool connected) {
  if (!client.connected()) {
    Serial.println("Not connected to MQTT, skipping send_connection_status");
    return;
  }
  DynamicJsonDocument doc(128);
  doc["device_id"] = device_id;
  doc["status"] = connected ? "connected" : "disconnected";
  char buffer[128];
  serializeJson(doc, buffer);
  if (client.publish(status_topic.c_str(), buffer)) {
    Serial.println("Connection status sent to " + status_topic + ": " + String(buffer));
  } else {
    Serial.println("Failed to send connection status to " + status_topic);
  }
}

void checkWiFiAndMQTT() {
  bool currentStatus = (WiFi.status() == WL_CONNECTED) && client.connected();
  if (currentStatus != previousConnectionStatus) {
    send_connection_status(currentStatus);
    previousConnectionStatus = currentStatus;
    Serial.println("Connection status changed: " + String(currentStatus ? "connected" : "disconnected"));
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  isConnected = WiFi.status() == WL_CONNECTED;
  previousConnectionStatus = WiFi.status() == WL_CONNECTED && client.connected();
  send_connection_status(previousConnectionStatus);
  if (isConnected) {
    reconnect();
  }
}

void loop() {
  if (!client.connected() && isConnected) reconnect();
  client.loop();
  checkWiFiAndMQTT();

  static unsigned long lastStatusUpdate = 0;
  if (millis() - lastStatusUpdate > 15000) {
    send_connection_status(WiFi.status() == WL_CONNECTED && client.connected());
    lastStatusUpdate = millis();
  }
}