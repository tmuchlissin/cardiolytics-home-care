from flask import Blueprint, Flask, render_template, redirect, url_for, flash
from app.forms import PatientDataForm
from app.models import db, PatientData
from app.utils import load_active_model


cvd_predict = Blueprint('cvd_predict', __name__)

@cvd_predict.route('/cvd_predict', methods=['GET', 'POST'])
def upload_menu():
    form = PatientDataForm()
    if form.validate_on_submit():
        # Ambil data input dari form
        age = form.age.data
        height = form.height.data
        weight = form.weight.data
        ap_hi = form.ap_hi.data
        ap_lo = form.ap_lo.data

        # Muat model aktif dari database
        model = load_active_model()
        if not model:
            flash('Model tidak tersedia. Silakan hubungi admin.', 'warning')
            return redirect(url_for('cvd_predict.upload_menu'))

        # Persiapkan fitur sesuai dengan urutan yang digunakan saat training model
        # Pastikan transformasi data (misalnya casting ke integer) sesuai dengan ekspektasi model Anda
        features = [
            age,
            int(form.gender.data),          # Misalnya, '1' untuk Male, '2' untuk Female
            height,
            weight,
            ap_hi,
            ap_lo,
            int(form.cholesterol.data),
            int(form.gluc.data),
            int(form.smoke.data),
            int(form.alco.data),
            int(form.active.data)
        ]

        # Lakukan prediksi dengan model
        y_pred = model.predict([features])[0]  # Misalnya, model mengembalikan 0 atau 1
        cardio_result = bool(y_pred)            # True jika at risk, False jika healthy

        # Buat instance PatientData dengan hasil prediksi
        new_patient = PatientData(
            age=age,
            height=height,
            weight=weight,
            gender=form.gender.data,
            ap_hi=ap_hi,
            ap_lo=ap_lo,
            cholesterol=int(form.cholesterol.data),
            gluc=int(form.gluc.data),
            smoke=bool(int(form.smoke.data)),
            alco=bool(int(form.alco.data)),
            active=bool(int(form.active.data)),
            cardio=cardio_result
        )

        # Simpan data ke database
        db.session.add(new_patient)
        db.session.commit()
        flash('Data berhasil diunggah dan prediksi telah dilakukan!', 'success')
        return redirect(url_for('cvd_predict.upload_menu'))
    
    return render_template('cvd_predict.html', form=form, navbar_title='CVD Predict')
