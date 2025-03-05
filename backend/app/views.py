
from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/index')
def index():
    return render_template('index.html', navbar_title='Home')
