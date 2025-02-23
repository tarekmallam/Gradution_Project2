from flask import Flask
import os
# import tensorflow as tf

def create_app():
    app = Flask(__name__)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SUPABASE_URI')
    # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app
