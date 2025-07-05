# ====================== KAFKA Settings ======================
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "bp_data"

# ====================== Blood Pressure API Endpoint ======================
BLOOD_PRESSURE_ENDPOINT = "http://localhost:5000/bp-monitor/api/blood_pressure"

# ====================== MQTT Settings ======================
MQTT_BROKER = "192.168.0.101"  # Change if needed, e.g., "192.168.1.200"
MQTT_PORT = 1883
MQTT_TOPIC = "bp_monitor/#"  # Wildcard to subscribe to all bp_monitor/BPXXXX topics
MQTT_USERNAME = "espuser"
MQTT_PASSWORD = "esp32mqtt!"
