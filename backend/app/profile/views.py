from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.models import PatientProfile, db
from app.forms import PatientProfileForm

profile = Blueprint('profile', __name__, url_prefix='/profile')

@profile.route('/edit', methods=['GET', 'POST'])
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
        return redirect(url_for('main.index'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {form[field].label.text}: {error}", 'danger')

    return render_template(
        'edit_profile.html',
        form=form,
        navbar_title='Profile'
    )