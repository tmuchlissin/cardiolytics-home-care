from functools import wraps
from flask import Blueprint, render_template, redirect, url_for, flash, abort, request
from flask_login import login_user, logout_user, current_user, login_required
from app.models import UserRole, User, Models
from app.extensions import db, mail 
from app.forms import EditUserForm, ModelForm
from sqlalchemy import or_

from werkzeug.utils import secure_filename

admin = Blueprint('admin', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != UserRole.admin:
            abort(403) 
        return f(*args, **kwargs)
    return decorated_function

@admin.route('/admin/user-approval')
@login_required
@admin_required
def user_approval():
    
    pending_users = User.query.filter(
        or_(User.approved == None, User.approved == False)
    ).all()
    return render_template('user_approval.html', navbar_title="User Approval", pending_users=pending_users)


@admin.route('/approve_user/<user_id>', methods=['POST'])
@login_required
@admin_required
def approve_user(user_id):
    user = User.query.get(user_id)
    if user:
        user.approved = True
        db.session.commit()
        #flash(f"User {user.user_name} has been approved.", "success")
    else:
        flash("❕User not found.", "danger")
    return redirect(url_for('admin.user_approval'))


@admin.route("/delete-user/<string:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for("admin.user_management"))


@admin.route('/admin/user-management')
@login_required
@admin_required
def user_management():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')

    # Query dasar hanya untuk user yang sudah disetujui
    query = User.query.filter(User.approved == True)

    # Jika ada pencarian, filter berdasarkan semua kolom yang tersedia
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

    return render_template(
        'user_management.html',
        navbar_title="Manage Users",
        users=users,
        pagination=pagination,
        entries_per_page=per_page,
        search_query=search_query  
    )


@admin.route('/admin/log-cvd-predict')
@login_required
@admin_required
def log_cvd_predict():
    return render_template('log_cvd_predict.html', navbar_title="Log Patient Data")

@admin.route('/reject_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reject_user(user_id):
    user = User.query.get_or_404(user_id)
    user.approved = False
    db.session.commit()
    #flash("User rejected.", "warning")
    return redirect(url_for('admin.user_approval'))  # Ganti sesuai dengan endpoint tampilan user approval

@admin.route("/admin/edit-user/<string:user_id>", methods=["GET", "POST"])
@login_required
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    form = EditUserForm(obj=user)  # Isi form dengan data user

    # Ambil pagination agar halaman tetap bisa dirender dengan benar
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
    
    if form.validate_on_submit():
        # Cek apakah email sudah digunakan oleh user lain
        existing_email = User.query.filter(User.email == form.email.data, User.id != user.id).first()
        if existing_email:
            flash("⚠️ Email already exists!", "warning")
            return render_template(
                    "user_management.html", 
                    form=form, 
                    user=user, 
                    users=users, 
                    pagination=pagination,  # ✅ Pastikan pagination selalu dikirim
                    entries_per_page=per_page, 
                    search_query=search_query
                )

        # Cek apakah phone_number sudah digunakan oleh user lain
        existing_phone = User.query.filter(User.phone_number == form.phone_number.data, User.id != user.id).first()
        if existing_phone:
            flash("⚠️ Phone number already exists!", "warning")
            return render_template(
                "user_management.html", 
                form=form, 
                user=user, 
                users=users, 
                pagination=pagination,  # ✅ Pastikan pagination selalu dikirim
                entries_per_page=per_page, 
                search_query=search_query
            )

        
        # # Cek apakah device_id sudah digunakan oleh user lain
        # existing_device = User.query.filter(User.device_id == form.device_id.data, User.id != user.id).first()
        # if existing_device:
        #     flash("⚠️ Device ID already exists!", "danger")
        #     return redirect(url_for("admin.edit_user", user_id=user_id))

        # Jika tidak ada duplikat, lanjutkan update data user
        user.full_name = form.full_name.data
        user.user_name = form.user_name.data
        user.email = form.email.data
        user.phone_number = form.phone_number.data
        user.role = form.role.data

        status = form.approved.data
        if status == "approved":
            user.approved = True
        elif status == "rejected":
            user.approved = False
        else:
            user.approved = None

        db.session.commit()
        flash("✅ User updated successfully!", "success")
        return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))

    
    return render_template(
        "user_management.html", 
        form=form, 
        user=user, 
        users=users, 
        pagination=pagination, 
        entries_per_page=per_page, 
        search_query=search_query
    )



@admin.route('/admin/settings')
@login_required
@admin_required
def settings():
    # Ambil model yang sudah diunggah (hanya satu yang diperbolehkan)
    model = Models.query.order_by(Models.created_at.desc()).first()
    form = ModelForm()  # Instance form untuk upload
    return render_template('admin_settings.html', model=model, form=form, navbar_title="Settings")


# @admin.route('/set_active_model/<int:model_id>', methods=['GET'])
# @login_required
# @admin_required
# def set_active_model(model_id):
#     model = Models.query.get_or_404(model_id)
    
#     # Nonaktifkan semua model yang aktif
#     Models.query.filter_by(is_active=True).update({'is_active': False})
    
#     # Set model yang dipilih menjadi aktif
#     model.is_active = True
#     db.session.commit()
    
#     flash('The model has been set as active!', 'success')
#     return redirect(url_for('admin.settings'))

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

    # Jika sudah ada model, tolak upload baru
    existing_model = Models.query.first()
    if existing_model:
        flash('A model already exists. Please delete it before uploading a new one.', 'warning')
        return redirect(url_for('admin.settings'))

    form = ModelForm()
    if form.validate_on_submit():
        file_data = form.filename.data  # FileField dari form
        if file_data:
            # Baca file sebagai binary dan ambil nama file yang aman
            model_binary = file_data.read()
            new_filename = secure_filename(file_data.filename)
            
            new_model = Models(
                name=form.name.data,
                filename=new_filename,
                file=model_binary,
                is_active=True,  # Model baru langsung di-set aktif
                user_id=current_user.id
            )
            db.session.add(new_model)
            db.session.commit()
            flash('✅ Model uploaded successfully!', 'success')
            return redirect(url_for('admin.settings'))
        else:
            flash('No file selected.', 'warning')
            return redirect(url_for('admin.settings'))
    
    # Jika GET atau validasi gagal:
    return render_template('admin_settings.html', model=None, form=form, navbar_title="Settings")




@admin.route('/admin/logout')
@login_required
@admin_required
def logout():
    logout_user()  # Menghapus sesi pengguna
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('auth.login'))  # Redirect ke halaman login


