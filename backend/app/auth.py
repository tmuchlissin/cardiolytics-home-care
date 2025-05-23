from flask import Blueprint, render_template, redirect, url_for, flash, abort, request, current_app
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db, mail 
from app.models import User, UserRole
from app.forms import RegistrationForm, LoginForm
import jsonify
from flask_mail import Mail, Message
from app.utils import generate_reset_token, verify_reset_token, send_email_async
import threading
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

auth = Blueprint('auth', __name__)

@auth.route('/', methods=['GET', 'POST'])
def landing_page():
     return render_template('landing_page.html')
    
@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(id=form.patient_id.data).first()
        if user and check_password_hash(user.password, form.password.data):
            # Periksa apakah user sudah di-approve oleh admin
            if user.approved is not True:
                flash("⏳ Your account is pending approval. Please wait for admin approval.", "attention")
                return redirect(url_for('auth.login'))
            # Jika sudah di-approve, login user
            login_user(user)
            
            # Debug
            print(f"✅ User {current_user.id} logged in with role {current_user.role}")
            
            if user.role == UserRole.admin:
                return redirect(url_for('admin.user_approval'))
            else:
                return redirect(url_for('main.index'))
        else:
            flash("❌ Invalid Patient ID or password.", "danger")
            print("❌ Invalid login attempt!")  # Debug
    return render_template('login.html', form=form)

@auth.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            token = generate_reset_token(email)
            reset_url = url_for('auth.reset_password', token=token, _external=True)
            subject = "Password Reset Instructions"
            body = f"Hi {user.user_name},\n\nTo reset your password, click the following link:\n{reset_url}\n\nIf you did not request a password reset, please ignore this email."
            
            msg = Message(subject, sender=os.getenv('MAIL_DEFAULT_SENDER'), recipients=[email])
            msg.body = body
            
            # Kirim email dalam thread dengan current_app
            thread = threading.Thread(target=send_email_async, args=(current_app._get_current_object(), msg))
            thread.start()

            flash("✅ Instructions to reset your password have been sent to your email.", "success")
            return redirect(url_for('auth.login'))
        else:
            flash("❕Email address not found.", "danger")

    return render_template('forgot_password.html')


@auth.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if not email:
        flash("❕The password reset link is invalid or has expired.", "danger")
        return redirect(url_for('auth.forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            flash("❕Passwords do not match.", "danger")
            return render_template('reset_password.html', token=token)
        if len(new_password) < 8:
            flash("❕Password must be at least 8 characters long.", "danger")
            return render_template('reset_password.html', token=token)
        
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("✅ Your password has been updated. Please log in.", "success")
            return redirect(url_for('auth.login'))
        else:
            flash("❕User not found.", "danger")
            return redirect(url_for('auth.forgot_password'))
    
    return render_template('reset_password.html', token=token)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    #print("🚀 register() function triggered")
    #print(f"Form Data Received: {form.data}")  # Debugging

    if form.validate_on_submit():
        print("✅ Form validation successful!")

        # Format phone number terlebih dahulu
        phone_number = form.phone_number.data
        if phone_number.startswith('0'):
            phone_number = '62' + phone_number[1:]

        existing_user = User.query.filter_by(id=form.patient_id.data).first()
        if existing_user:
            flash('⚠ Patient ID is already registered. Please use a different one.', 'warning')
            return redirect(url_for('auth.register'))

        existing_email = User.query.filter_by(email=form.email.data).first()
        if existing_email:
            flash('⚠ Email is already registered. Please use a different one.', 'warning')
            return redirect(url_for('auth.register'))

        existing_phone = User.query.filter_by(phone_number=phone_number).first()
        if existing_phone:
            flash('⚠ Phone number is already registered. Please use a different one.', 'warning')
            return redirect(url_for('auth.register'))

        hashed_password = generate_password_hash(form.password.data)

        new_user = User(
            id=form.patient_id.data,
            full_name=form.full_name.data,
            user_name=form.nickname.data,
            email=form.email.data,
            phone_number=phone_number,
            password=hashed_password,
            role=UserRole.user
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            print("✅ User successfully stored in database!")  # Debugging
            flash('✅ Registration successful! Please sign in.', 'success')
            return redirect(url_for('auth.register'))
        except Exception as e:
            db.session.rollback()
            print(f"❌ Database error: {e}")  # Debugging
            flash('❌ An error occurred while registering. Please try again.', 'danger')

    else:
        # print("❌ Form validation failed!")  # Debugging
        for field, errors in form.errors.items():
            for error in errors:
                print(f"❌ Error in {field}: {error}")  # Debugging

    return render_template('register.html', form=form)

@auth.route('/logout')
@login_required
def logout():

    if current_user.role == UserRole.user:
        from app.cardiobot.views import clear_cache
        clear_cache()
    logout_user()
    return redirect(url_for('auth.login'))

@auth.route('/check-patient-id/<patient_id>', methods=['GET'])
def check_patient_id(patient_id):
    """
    Check if the given patient_id exists in the database.
    If the patient_id exists but all other fields (name, email, phone, role) are empty, return valid.
    Otherwise, return false.
    """
    user = User.query.filter_by(patient_id=patient_id).first()
    
    print(f"Checking patient_id: {patient_id}")
    
    if user:
        print(f"User found: {user.patient_id}")
    else:
        print("User is None")
    
    if user and all(
        getattr(user, field, None) in [None, '']
        for field in ["full_name", "email", "user_name", "role"]
    ):
        return jsonify({'valid': True})  
    
    return jsonify({'valid': False})  