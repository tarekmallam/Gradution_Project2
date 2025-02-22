from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import psycopg2
from functools import wraps
import os
from PIL import Image
import numpy as np
import tensorflow as tf

auth = Blueprint('auth', __name__)

# Environment-based database connection
def create_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        dbname=os.getenv('DB_NAME', 'skin_cancer'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '1172002'),
        port=os.getenv('DB_PORT', '5432')
    )

# Load the model
model = tf.keras.models.load_model('skin_cancer_classifier_model.h5')

# Class labels
CLASSES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image_path):
    """Process image for prediction"""
    img = Image.open(image_path)
    img = img.resize((100, 75))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    return img_array



# Decorator for admin-only routes
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/')
def home():
    return render_template("index.html")

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            with create_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                    user = cursor.fetchone()

                    if user:
                        user_id, user_email, hashed_password, is_admin = user
                        if check_password_hash(hashed_password, password):
                            session['user_id'] = user_id
                            session['email'] = user_email
                            session['is_admin'] = is_admin
                            flash('Login successful!', 'success')
                            return redirect(url_for('auth.home'))
                        else:
                            flash('Invalid email or password.', 'danger')
                    else:
                        flash('Invalid email or password.', 'danger')
        except Exception as e:
            print(f"Error: {e}")
            flash('An error occurred. Please try again.', 'danger')

    return render_template('login.html')

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if len(email) < 4:
            flash('Email must be greater than 4 characters.', 'error')
        elif password != confirm_password:
            flash('Passwords don\'t match.', 'error')
        elif len(password) < 7:
            flash('Password must be at least 7 characters.', 'error')
        else:
            try:
                with create_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                        if cursor.fetchone():
                            flash('Email already registered. Please log in.', 'error')
                        else:
                            hashed_password = generate_password_hash(password, method='sha256')
                            cursor.execute(
                                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                                (name, email, hashed_password)
                            )
                            conn.commit()
                            flash('Account created successfully!', 'success')
                            return render_template("index.html")
            except Exception as e:
                print(f"Error: {e}")
                flash('An error occurred. Please try again.', 'danger')

    return render_template('signup.html')

@auth.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/forgot')
def forgot():
    return redirect(url_for('auth.login'))

@auth.route('/manage_users')
@admin_required
def manage_users():
    try:
        with create_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, name, email FROM users")
                users = cursor.fetchall()
        return render_template('manage_users.html', users=users)
    except Exception as e:
        print(f"Error: {e}")
        flash('Failed to retrieve users.', 'danger')
        return redirect(url_for('auth.home'))

@auth.route('/profile')
def profile():
    return render_template('profile.html')

@auth.route('/about')
def about():
    return render_template('about-us.html')

@auth.route('/contact')
def contact_us():
    return render_template('contact.html')

@auth.route('/blog')
def blog():
    return render_template('blog.html')

@auth.route('/detect')
def detect():
    return render_template('detecth.html')

@auth.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(auth.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image and make prediction
        img_array = process_image(filepath)
        predictions = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]) * 100)
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'image_path': f'uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@auth.route('/add_topic', methods=['GET', 'POST'])
def add_topic():
    # if request.method == 'POST':
    #     title = request.form['title']
    #     description = request.form['description']
    #     # Handle form inputs and save to DB
    #     flash('Topic added successfully!', 'success')
    #     return redirect(url_for('auth.home'))
    return render_template('add_topic.html')