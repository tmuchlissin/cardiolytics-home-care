from flask import Blueprint, Flask, render_template, redirect, url_for, flash
from app.forms import PatientDataForm
from app.models import db, PatientData

cvd_predict = Blueprint('cvd_predict', __name__)

@cvd_predict.route('/cvd_predict', methods=['GET', 'POST'])
def upload_menu():
    form = PatientDataForm()
    if form.validate_on_submit():
        new_patient = PatientData(
            age=form.age.data,
            height=form.height.data,
            weight=form.weight.data,
            gender=form.gender.data,
            ap_hi=form.ap_hi.data,
            ap_lo=form.ap_lo.data,
            cholesterol=form.cholesterol.data,
            gluc=form.gluc.data,
            smoke=form.smoke.data,
            alco=form.alco.data,
            active=form.active.data,
            cardio=form.cardio.data
        )
        db.session.add(new_patient)
        db.session.commit()
        flash('Data uploaded successfully!', 'success')
        return redirect(url_for('upload'))
    return render_template('cvd_predict.html', form=form)