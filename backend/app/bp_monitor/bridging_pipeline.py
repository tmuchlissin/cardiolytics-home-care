import threading
import logging
import time

import emqx_to_kafka
import kafka_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridging_pipeline")

def run_emqx_kafka_thread():
    try:
        logger.info("[PIPELINE] Starting emqx_to_kafka...")
        emqx_to_kafka.main()
    except Exception as e:
        logger.error(f"[PIPELINE] emqx_to_kafka crashed: {e}")

def run_kafka_to_db_thread():
    try:
        logger.info("[PIPELINE] Starting kafka_to_db...")
        kafka_to_db.main()
    except Exception as e:
        logger.error(f"[PIPELINE] kafka_to_db crashed: {e}")

if __name__ == "__main__":
    t1 = threading.Thread(target=run_emqx_kafka_thread)
    t2 = threading.Thread(target=run_kafka_to_db_thread)

    t1.start()
    time.sleep(2) 
    t2.start()

    t1.join()
    t2.join()
