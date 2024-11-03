from flask import Blueprint, Flask, render_template, redirect, url_for, flash
from app.models import db

cardiobot = Blueprint('cardiobot', __name__)

@cardiobot.route('/cardiobot')
def chat():
    return render_template('cardiobot.html', navbar_title='CVD Predict')
