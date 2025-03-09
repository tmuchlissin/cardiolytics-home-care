from functools import wraps
from flask import Blueprint, render_template, redirect, url_for, flash, abort, request,jsonify
from flask_login import login_user, logout_user, current_user, login_required
from app.models import UserRole, User, Models, Device
from app.extensions import db, mail 
from app.forms import EditUserForm, ModelForm, DeviceForm
from sqlalchemy import or_
from werkzeug.utils import secure_filename

admin = Blueprint('admin', __name__)


###########################################################
######################### AUTH ############################
###########################################################
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != UserRole.admin:
            abort(403) 
        return f(*args, **kwargs)
    return decorated_function

@admin.route('/admin/logout')
@login_required
@admin_required
def logout():
    logout_user() 
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('auth.login'))

###########################################################
####################### USER APPROVAL #####################
###########################################################
@admin.route('/admin/user-approval')
@login_required
@admin_required
def user_approval():
    pending_users = User.query.filter(
        or_(User.approved == None, User.approved == False)
    ).all()
    return render_template('user_approval.html', navbar_title="User Approval", pending_users=pending_users)

@admin.route('/admin/approve-user/<user_id>', methods=['POST'])
@login_required
@admin_required
def approve_user(user_id):
    user = User.query.get(user_id)
    if user:
        user.approved = True
        db.session.commit()
        print(f"User {user.user_name} has been approved.", "success")
    else:
        print("❕User not found.", "danger")
    return redirect(url_for('admin.user_approval'))

###########################################################
##################### MANAGE USER #########################
@admin.route('/admin/user-management')
@login_required
def user_management():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    query = User.query.filter(User.approved == True)
    if search_query:
        search_pattern = f"%{search_query}%"
        query = query.filter(
            (User.id.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern)) |
            (User.user_name.ilike(search_pattern)) |
            (User.email.ilike(search_pattern)) |
            (User.phone_number.ilike(search_pattern)) |
            (User.role.ilike(search_pattern)) |
            (User.approved.ilike(search_pattern))
        )
    
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    users = pagination.items

    devices = Device.query.order_by(Device.id).all()
    assigned_device_ids = { str(u.device_id) for u in User.query.filter(User.device_id.isnot(None)).all() }
    available_devices = [d for d in devices if str(d.id) not in assigned_device_ids]

    device_choices = [('none', 'None')] + [(str(d.id), f"{d.id}") for d in available_devices]

    form = EditUserForm()
    form.device_id.choices = device_choices

    return render_template(
        'user_management.html',
        navbar_title="Manage Users",
        users=users,
        devices=devices,
        form=form,  # pastikan form dikirim
        pagination=pagination,
        entries_per_page=per_page,
        search_query=search_query  
    )

@admin.route("/admin/user-management/edit-user/<string:user_id>", methods=["GET", "POST"])
@login_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    query = User.query.filter(User.approved == True)
    if search_query:
        search_pattern = f"%{search_query}%"
        query = query.filter(
            (User.id.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern)) |
            (User.user_name.ilike(search_pattern)) |
            (User.email.ilike(search_pattern)) |
            (User.phone_number.ilike(search_pattern)) |
            (User.role.ilike(search_pattern)) |
            (User.approved.ilike(search_pattern))
        )
    
    devices = Device.query.order_by(Device.id).all()
    device_choices = [('none', 'None')] + [(d.id, f"{d.id}") for d in devices]

    if request.method == "POST":
        full_name    = request.form.get("full_name")
        user_name    = request.form.get("user_name")
        email        = request.form.get("email")
        phone_number = request.form.get("phone_number")
        role         = request.form.get("role")
        approved     = request.form.get("approved")
        device_id    = request.form.get("device_id")

        existing_email = User.query.filter(User.email == email, User.id != user.id).first()
        if existing_email:
            flash("⚠️ Email already exists!", "warning")
            return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
        
        existing_phone = User.query.filter(User.phone_number == phone_number, User.id != user.id).first()
        if existing_phone:
            flash("⚠️ Phone number already exists!", "warning")
            return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
        
        user.full_name    = full_name
        user.user_name    = user_name
        user.email        = email
        user.phone_number = phone_number
        user.role         = role
        
        if approved == "approved":
            user.approved = True
        elif approved == "rejected":
            user.approved = False
        else:
            user.approved = None
        
        if device_id in ("none", ""):
            user.device_id = None
        else:
            user.device_id = device_id
        
        db.session.commit()
        flash("✅ User updated successfully!", "success")
        return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
    
    return render_template(
        "user_management.html",
        user=user,
        devices=devices,
        device_choices=device_choices,
        page=page, 
        per_page=per_page, 
        search=search_query
    )

@admin.route("/admin/user-management/delete-user/<string:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    if user:
        db.session.delete(user)
        db.session.commit()
        flash("✅ User deleted successfully!", "success")
    
    return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))


###########################################################
####################### PATIENT DATA ######################
###########################################################
@admin.route('/admin/log-cvd-predict')
@login_required
@admin_required
def log_cvd_predict():
    return render_template('log_cvd_predict.html', navbar_title="Log Patient Data")


###########################################################
######################### SETTINGS ########################
###########################################################
@admin.route('/admin/settings')
@login_required
@admin_required
def settings():
    active_tab = request.args.get('tab', 'models') 
    
    model = Models.query.order_by(Models.created_at.desc()).first()

    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '').strip()
    
    device_query = Device.query
    
    if search_query:
        device_query = device_query.filter(
            or_(
                Device.id.ilike(f"%{search_query}%"),
                Device.model.ilike(f"%{search_query}%")
            )
        )
    
    device_pagination = device_query.order_by(Device.registered_at.desc()).paginate(page=page, per_page=5, error_out=False)
    devices = device_pagination.items
    
    form = ModelForm()
    return render_template(
        'admin_settings.html',
        model=model,
        devices=devices,
        device_pagination=device_pagination,
        form=form,
        search_query=search_query,
        navbar_title="Settings",
        active_tab=active_tab
    )

#------------------------- Model --------------------------#

@admin.route('/delete_model/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def delete_model(model_id):
    model = Models.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    flash('✅ Model deleted successfully!', 'success')
    return redirect(url_for('admin.settings'))

@admin.route('/upload_model', methods=['GET', 'POST'])
@login_required
def upload_model():
    if current_user.role != UserRole.admin:
        flash('You are not authorized to upload models.', 'danger')
        return redirect(url_for('index'))

    existing_model = Models.query.first()
    if existing_model:
        flash('⚠️ A model already exists. Please delete it before uploading a new one.', 'warning')
        return redirect(url_for('admin.settings'))

    form = ModelForm()
    if form.validate_on_submit():
        file_data = form.filename.data  
        if file_data:
            model_binary = file_data.read()
            new_filename = secure_filename(file_data.filename)
            new_model = Models(
                name=form.name.data,
                filename=new_filename,
                file=model_binary,
                is_active=True,  
                user_id=current_user.id
            )
            db.session.add(new_model)
            db.session.commit()
            flash('✅ Model uploaded successfully!', 'success')
            return redirect(url_for('admin.settings'))
        else:
            flash('No file selected.', 'warning')
            return redirect(url_for('admin.settings', tab='models'))
    
    return render_template('admin_settings.html', tab='models', model=None, form=form, navbar_title="Settings")

#------------------------ Device --------------------------#

@admin.route('/admin/upload_device', methods=['GET', 'POST'])
@login_required
@admin_required
def upload_device():
    form = DeviceForm()  
    if form.validate_on_submit():
        device_id = form.device_id.data
        model = form.model.data
        
        if Device.query.filter_by(id=device_id).first():
            flash('⚠️ A device with this ID is already registered!', 'warning')
            return redirect(url_for('admin.settings', tab='devices'))
        
        new_device = Device(id=device_id, model=model)
        db.session.add(new_device)
        db.session.commit()
        flash('✅ Device successfully added', 'success')
        return redirect(url_for('admin.settings', tab='devices'))
        
    return render_template('admin_settings.html', tab='devices', form=form, navbar_title="Settings")

@admin.route('/delete_device/<string:device_id>', methods=['POST'])
@login_required
@admin_required
def delete_device(device_id):
    device = Device.query.get_or_404(device_id)
    db.session.delete(device)
    db.session.commit()
    flash('✅ Device deleted successfully!', 'success')
    return redirect(url_for('admin.settings', tab='devices'))

@admin.route('/edit_device', methods=['POST'])
@login_required
@admin_required
def edit_device():
    device_id = request.form.get('device_id')
    dev = Device.query.get_or_404(device_id)

    new_device_id = request.form.get('edit_device_id')
    new_model = request.form.get('edit_model')

    if Device.query.filter(Device.id == new_device_id, Device.id != dev.id).first():
        flash("⚠️ Device ID already in use!", "danger")
        return redirect(url_for('admin.settings', tab='devices'))

    dev.id = new_device_id
    dev.model = new_model
    db.session.commit()

    flash("✅ Device updated successfully!", "success")
    return redirect(url_for('admin.settings', tab='devices'))

#------------------------- Docs --------------------------#