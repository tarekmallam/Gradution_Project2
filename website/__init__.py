from flask import Flask
import os
import secrets
# import tensorflow as tf

def create_app():
    app = Flask(__name__)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 16MB max file size

    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_key')

    # ✅ Configure session to use filesystem instead of cookies
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False  # Keep session active
    app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # Session lasts for 1 day


    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app
