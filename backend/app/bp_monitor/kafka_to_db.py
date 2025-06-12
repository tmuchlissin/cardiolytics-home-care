import requests
from kafka import KafkaConsumer
import json
import time
import logging
from flask import current_app

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka settings
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "bp_data"

# BP endpoint
BLOOD_PRESSURE_ENDPOINT = f"http://localhost:5000/bp-monitor/api/blood_pressure"

def main():
    max_retries = 5
    retry_count = 0
    consumer = None
    while retry_count < max_retries and consumer is None:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            logger.info("Connected to Kafka")
            break
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}. Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(5)
    else:
        logger.error("Failed to connect to Kafka after max retries")
        return

    for message in consumer:
        data = message.value
        logger.info(f"Consumed from Kafka: {data}")
        try:
            response = requests.post(
                BLOOD_PRESSURE_ENDPOINT,
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                logger.info(f"Data sent to Flask: {response.json()}")
            else:
                logger.error(f"Failed to send data to Flask: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error sending data to Flask: {e}")

if __name__ == "__main__":
    main()
