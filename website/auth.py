from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from supabase import create_client, Client  # Supabase SDK
from dotenv import load_dotenv
import psycopg2
import uuid
from datetime import datetime
import secrets


load_dotenv()
auth = Blueprint('auth', __name__)

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Successfully connected to the database!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")

SUPABASE_URL = os.getenv("SUPABASE_URI")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

if SUPABASE_URL and SUPABASE_API_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    print("✅ Successfully connected to Supabase!")
else:
    print("❌ Missing Supabase credentials. Check your .env file.")

# Class labels for image classification
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
# def admin_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if not session.get('is_admin'):
#             flash('You do not have permission to access this page.', 'danger')
#             return redirect(url_for('auth.login'))
#         return f(*args, **kwargs)
#     return decorated_function

@auth.route('/')
def home():
    return render_template("index.html")

# 🟢 Login Route
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            response = supabase.table('user').select('*').eq('email', email).single().execute()
            user = response.data
            
            if user and check_password_hash(user['password'], password):
                session.clear()
                session.permanent = True
                session['userid'] = user['userID']
                session['email'] = user['email']
                session['role'] = user.get('role', 'user')

                flash('✅ Login successful!', 'success')
                return redirect(url_for('dashboard' if user['role'] == 'admin' else 'main.home'))
            else:
                flash('❌ Invalid email or password.', 'danger')
        except Exception as e:
            flash('⚠ An error occurred during login. Please try again.', 'danger')

    return render_template('login.html')

# 🔵 Signup Route
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # ✅ Get form data and strip any leading/trailing spaces
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # ✅ Validate inputs to avoid NoneType errors
        if not email or not password or not confirm_password:
            flash('⚠ All fields are required.', 'error')
            return redirect(url_for('auth.signup'))
        
        if len(email) < 4:
            flash('⚠ Email must be at least 4 characters long.', 'error')
            return redirect(url_for('auth.signup'))

        if password != confirm_password:
            flash('⚠ Passwords do not match.', 'error')
            return redirect(url_for('auth.signup'))

        if len(password) < 7:
            flash('⚠ Password must be at least 7 characters long.', 'error')
            return redirect(url_for('auth.signup'))

        try:
            # ✅ Check if the email is already registered
            response = supabase.table('users').select('email').eq('email', email).single().execute()
            print("Response from DB:", response)  # Debugging line
            
            if response.data:
                flash('⚠ This email is already registered.', 'error')
                return redirect(url_for('auth.signup'))

            # ✅ Hash the password for security
            hashed_password = generate_password_hash(password, method='sha256')

            new_user = {
                "userID": str(uuid.uuid4()),
                "email": email,
                "password": hashed_password,
                "username": name,
                "role": "user",
                "registrationDate": datetime.utcnow().isoformat()
            }

            # ✅ Insert new user into the database
            insert_response = supabase.table('users').insert(new_user).execute()
            print("Insert response:", insert_response)  # Debugging line

            flash('🎉 Account created successfully!', 'success')
            print("Redirecting to login page")  # Debugging line
            return redirect(url_for('auth.login'))  # ✅ Redirect after successful signup

        except Exception as e:
            flash(f'⚠ An error occurred during signup: {str(e)}', 'danger')
            print("Signup error:", str(e))  # Debugging line

    return render_template('signup.html')


@auth.route('/logout')
def logout():
    session.clear()
    flash('✅ Logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@auth.route('/forgot')
def forgot():
    return redirect(url_for('auth.login'))

# @auth.route('/manage_users')
# @admin_required
# def manage_users():
#     try:
#         with create_connection() as conn:
#             with conn.cursor() as cursor:
#                 cursor.execute("SELECT id, name, email FROM users")
#                 users = cursor.fetchall()
#         return render_template('manage_users.html', users=users)
#     except Exception as e:
#         print(f"Error: {e}")
#         flash('Failed to retrieve users.', 'danger')
#         return redirect(url_for('auth.home'))

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

@auth.route('/ak')
def ak():
    return render_template('ak.html')

@auth.route('/bbc')
def bbc():
    return render_template('bbc.html')

@auth.route('/bkl')
def bkl():
    return render_template('bkl.html')

@auth.route('/df')
def df():
    return render_template('df.html')

@auth.route('/nv')
def nv():
    return render_template('nv.html')

@auth.route('/scc')
def scc():
    return render_template('scc.html')

@auth.route('/vl')
def vl():
    return render_template('vl.html')

@auth.route('/melanoma')
def melanoma():
    return render_template('melanoma.html')

@auth.route('/detect')
def detect():
    return render_template('detect.html')

@auth.route('/predict', methods=['POST'])
def predict():
    return render_template('result.html')


# @auth.route('/predict', methods=['POST'])
# def predict():source GP_2025/bin/activate
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(auth.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Process image and make prediction
#         img_array = process_image(filepath)
#         predictions = model.predict(img_array)
#         predicted_class = CLASSES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]) * 100)
        
#         return jsonify({
#             'class': predicted_class,
#             'confidence': confidence,
#             'image_path': f'uploads/{filename}'
#         })
    
#     return jsonify({'error': 'Invalid file type'}), 400


@auth.route('/add_topic', methods=['GET', 'POST'])
def add_topic():
    # if request.method == 'POST':
    #     title = request.form['title']
    #     description = request.form['description']
    #     # Handle form inputs and save to DB
    #     flash('Topic added successfully!', 'success')
    #     return redirect(url_for('auth.home'))
    return render_template('add_topic.html')