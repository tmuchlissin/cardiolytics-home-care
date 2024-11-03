from flask import Blueprint, Flask, render_template, redirect, url_for, flash
from app.models import db

bp_monitor = Blueprint('bp_monitor', __name__)

@bp_monitor.route('/bp-monitor')
def monitor():
    return render_template('bp_monitor.html', navbar_title='BP Monitor')