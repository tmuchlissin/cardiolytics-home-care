from flask import Blueprint, render_template, redirect, url_for, flash, jsonify, request
from flask_login import login_required, current_user
from app.models import db, Device, BloodPressureRecord, User
from datetime import datetime
import uuid
import logging
from pytz import timezone, UTC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp_monitor = Blueprint('bp_monitor', __name__, url_prefix='/bp-monitor')

def parse_iso_to_wib(timestamp_str):
    utc_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    wib = timezone('Asia/Jakarta')
    return utc_dt.astimezone(wib)

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
    
    if device.status == 'disconnected':
        return jsonify({'status': 'error', 'message': 'Your device is disconnected.', 'connection': 'disconnected'})
    
    return jsonify({'status': 'success', 'connection': 'connected'})

@bp_monitor.route('/start_measurement', methods=['POST'])
@login_required
def start_measurement():
    device = Device.query.get(current_user.device_id)
    if not device or device.status != 'connected':
        return jsonify({'status': 'error', 'message': 'Device not connected'})
    return jsonify({'status': 'success', 'device_id': current_user.device_id})

@bp_monitor.route('/api/blood_pressure', methods=['POST'])
def receive_blood_pressure():
    data = request.get_json()
    logger.info(f"Received blood pressure data: {data}") 
    device_id = data.get('device_id')
    device = Device.query.get(device_id)
    if not device or device.status != 'connected':
        return jsonify({'status': 'error', 'message': 'Device not connected'}), 403
    user = User.query.filter_by(device_id=device_id).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
            
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
    return jsonify({'status': 'success'}), 200

@bp_monitor.route('/api/get_blood_pressure', methods=['GET'])
@login_required
def get_blood_pressure():
    recent_records = (
        BloodPressureRecord.query
        .filter_by(device_id=current_user.device_id)
        .order_by(BloodPressureRecord.timestamp.desc())
        .limit(30)
        .all()
    )
    if recent_records:
        wib = timezone('Asia/Jakarta')
        data = [
            {
                'systolic': record.systolic,
                'diastolic': record.diastolic,
                'pulse_rate': record.pulse_rate,
                'timestamp': record.timestamp.astimezone(wib).strftime('%d-%m-%Y, %H:%M:%S WIB')
            }
            for record in recent_records
        ]
        return jsonify(data)
    return jsonify([]), 200