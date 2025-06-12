import paho.mqtt.client as mqtt
from kafka import KafkaProducer
import json
import time
import logging
from flask import current_app

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MQTT settings
MQTT_BROKER = "192.168.0.111"
# MQTT_BROKER = "192.168.1.200"
MQTT_PORT = 1883
MQTT_TOPIC = "bp_monitor/BP0001"
MQTT_USERNAME = "espuser" 
MQTT_PASSWORD = "esp32mqtt!"

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "bp_data"

try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    logger.info("Connected to Kafka broker")
except Exception as e:
    logger.error(f"Failed to connect to Kafka: {e}")
    exit(1)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT broker with result code {rc}")
        client.subscribe(MQTT_TOPIC)
        logger.info(f"Subscribed to {MQTT_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT broker with result code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        logger.info(f"Received message on {msg.topic}: {payload}")
        data = json.loads(payload)
        producer.send(KAFKA_TOPIC, data)
        producer.flush()
        logger.info(f"Sent to Kafka topic {KAFKA_TOPIC}: {data}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def main():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            logger.info("Connected to MQTT broker")
            break
        except Exception as e:
            logger.error(f"Connection failed: {e}. Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(5)
    else:
        logger.error("Failed to connect to MQTT broker after max retries")
        return

    client.loop_forever()

if __name__ == "__main__":
    main()
