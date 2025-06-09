from flask import Blueprint, render_template, redirect, url_for, flash, jsonify, request, current_app
from flask_login import login_required, current_user
from app.models import db, Device, BloodPressureRecord, User
from datetime import datetime
import uuid
import logging
from pytz import timezone, UTC
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import json
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp_monitor = Blueprint('bp_monitor', __name__, url_prefix='/bp-monitor')

MQTT_BROKER = "192.168.0.111"
MQTT_PORT = 1883
MQTT_USERNAME = "espuser"
MQTT_PASSWORD = "esp32mqtt!"

last_status_cache = {}

def parse_iso_to_wib(timestamp_str):
    utc_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    wib = timezone('Asia/Jakarta')
    return utc_dt.astimezone(wib)

def update_device_status(device_id, status):
    try:
        if last_status_cache.get(device_id) == status:
            return 

        device = Device.query.get(device_id)
        if not device:
            logger.warning(f"Device {device_id} not found")
            return

        if device.status != status:
            device.status = status
            device.updated_at = datetime.utcnow()
            db.session.commit()
            logger.info(f"Updated device {device_id} status to {status}")
        else:
            logger.info(f"Device {device_id} status unchanged: still {status}")

        last_status_cache[device_id] = status

    except Exception as e:
        logger.error(f"Error updating device status: {e}")
        db.session.rollback()

def setup_mqtt_status_listener(app):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT for status updates")
            client.subscribe("bp_monitor/status/+")
        else:
            logger.error(f"Failed to connect to MQTT for status updates: {rc}")

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            device_id = data.get('device_id')
            status = data.get('status')

            if last_status_cache.get(device_id) == status:
                return

            logger.info(f"Raw MQTT status message on {msg.topic}: {payload}")
            logger.info(f"Parsed status update: {device_id} - {status}")

            with app.app_context():
                update_device_status(device_id, status)

        except Exception as e:
            logger.error(f"Error processing MQTT status message: {e}")

    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        logger.info("MQTT status listener started")
    except Exception as e:
        logger.error(f"Failed to connect to MQTT for status updates: {e}")

@bp_monitor.route('/dashboard')
@login_required
def monitor():
    data = BloodPressureRecord.query\
        .filter_by(device_id=current_user.device_id)\
        .order_by(BloodPressureRecord.timestamp.desc())\
        .first()
    if not current_user.device_id:
        flash('⏳ Device not set, please contact admin.', 'attention')
    return render_template('bp_monitor.html', navbar_title='BP Monitor', data=data, user=current_user)

@bp_monitor.route('/check_connection', methods=['POST'])
@login_required
def check_connection():
    if not current_user.device_id:
        return jsonify({'status': 'error', 'message': 'Device not set, please contact admin.', 'connection': 'none'})
    device = Device.query.get(current_user.device_id)
    if not device:
        return jsonify({'status': 'error', 'message': 'Device not found', 'connection': 'none'})
    logger.info(f"Checked connection for device {current_user.device_id}: {device.status}")
    return jsonify({'status': 'success', 'connection': device.status})

@bp_monitor.route('/start_measurement', methods=['POST'])
@login_required
def start_measurement():
    device = Device.query.get(current_user.device_id)
    if not device or device.status != 'connected':
        logger.error(f"Device {current_user.device_id} not connected or not found")
        return jsonify({'status': 'error', 'message': 'Device not connected'}), 403
    try:
        mqtt_topic = f"bp_monitor/command/{current_user.device_id}"
        payload = {"command": "start_measurement"}
        publish.single(
            mqtt_topic,
            payload=json.dumps(payload),
            hostname=MQTT_BROKER,
            port=MQTT_PORT,
            auth={'username': MQTT_USERNAME, 'password': MQTT_PASSWORD}
        )
        logger.info(f"Sent command to ESP32: {mqtt_topic} - {payload}")
        return jsonify({'status': 'success', 'device_id': current_user.device_id})
    except Exception as e:
        logger.error(f"Failed to send command to ESP32: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to start measurement'}), 500

@bp_monitor.route('/api/blood_pressure', methods=['POST'])
def receive_blood_pressure():
    data = request.get_json()
    logger.info(f"Received blood pressure data: {data}")
    device_id = data.get('device_id')
    device = Device.query.get(device_id)
    if not device or device.status != 'connected':
        logger.error(f"Device {device_id} not connected or not found")
        return jsonify({'status': 'error', 'message': 'Device not connected'}), 403
    user = User.query.filter_by(device_id=device_id).first()
    if not user:
        logger.error(f"User not found for device {device_id}")
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    try:
        bp_data = BloodPressureRecord(
            id=str(uuid.uuid4()),
            device_id=device_id,
            systolic=data['systolic'],
            diastolic=data['diastolic'],
            pulse_rate=data['pulse'],
            timestamp=parse_iso_to_wib(data['timestamp'])
        )
        db.session.add(bp_data)
        db.session.commit()
        logger.info(f"Stored blood pressure data for device {device_id}")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Error storing blood pressure data: {e}")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to store data'}), 500

@bp_monitor.route('/api/update_status', methods=['POST'])
def update_status():
    data = request.get_json()
    logger.info(f"Received status update via API: {data}")
    device_id = data.get('device_id')
    status = data.get('status')
    update_device_status(device_id, status)
    return jsonify({'status': 'success'}), 200

@bp_monitor.route('/api/get_blood_pressure', methods=['GET'])
@login_required
def get_blood_pressure():
    recent_records = BloodPressureRecord.query\
        .filter_by(device_id=current_user.device_id)\
        .order_by(BloodPressureRecord.timestamp.desc())\
        .limit(30)\
        .all()
    if recent_records:
        wib = timezone('Asia/Jakarta')
        data = [{'systolic': r.systolic, 'diastolic': r.diastolic, 'pulse_rate': r.pulse_rate, 'timestamp': r.timestamp.astimezone(wib).strftime('%d-%m-%Y, %H:%M:%S WIB')} for r in recent_records]
        logger.info(f"Retrieved {len(data)} records for device {current_user.device_id}")
        return jsonify(data)
    logger.info(f"No records found for device {current_user.device_id}")
    return jsonify([]), 200
