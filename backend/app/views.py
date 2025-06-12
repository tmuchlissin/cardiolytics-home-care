from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import current_user, login_required
from app.models import PatientProfile, PatientData, BloodPressureRecord, db
from datetime import datetime
from app.forms import PatientProfileForm

main = Blueprint('main', __name__, url_prefix='/main')

@main.route('/home')
@login_required
def home():
    profile = PatientProfile.query.filter_by(user_id=current_user.id).first()

    if not profile or not all([profile.date_of_birth, profile.gender, profile.national_id, profile.emergency_contact]):
        flash('⚠️ Please complete your profile before using the app features.', 'warning')
        return redirect(url_for('main.edit_profile')) 

    latest_patient_data = PatientData.query.filter_by(user_id=current_user.id)\
                                        .order_by(PatientData.submitted_at.desc()).first()
                                        
    latest_bp_record = None
    if current_user.device:
        latest_bp_record = BloodPressureRecord.query.filter_by(device_id=current_user.device.id)\
                                                    .order_by(BloodPressureRecord.timestamp.desc()).first()
    age = None
    if profile.date_of_birth:
        today = datetime.today()
        age = today.year - profile.date_of_birth.year - (
            (today.month, today.day) < (profile.date_of_birth.month, profile.date_of_birth.day)
        )

    return render_template(
        'main/home.html',
        navbar_title='Home',
        profile=profile,
        age=age,
        latest_patient_data=latest_patient_data,
        latest_bp_record=latest_bp_record,
        device_status=current_user.device.status if current_user.device else 'No device connected'
    )


@main.route('/home/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user_profile = PatientProfile.query.filter_by(user_id=current_user.id).first()
    form = PatientProfileForm(obj=user_profile)

    if form.validate_on_submit():
        if not user_profile:
            user_profile = PatientProfile(user_id=current_user.id)
        
        form.populate_obj(user_profile)
        db.session.add(user_profile)
        db.session.commit()
        flash('✅ Profil has been updated.', 'success')
        return redirect(url_for('main.home'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {form[field].label.text}: {error}", 'danger')

    return render_template(
        'main/edit_profile.html',
        form=form,
        navbar_title='Edit Profile'
    )