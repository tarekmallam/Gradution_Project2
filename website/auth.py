from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory, make_response
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
import mysql.connector
from mysql.connector import Error
from io import BytesIO
import base64
import traceback

from ctransformers import AutoModelForCausalLM
import logging
import threading
import mimetypes

auth = Blueprint('auth', __name__)

# Load environment variables from .env
load_dotenv()

# Load Supabase credentials and strip whitespace
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# # Debug: Print loaded environment variables (remove in production)
# print(f"Loaded SUPABASE_URL: {SUPABASE_URL} (type: {type(SUPABASE_URL)}, len: {len(SUPABASE_URL)})")
# print(f"Loaded SUPABASE_API_KEY: {SUPABASE_API_KEY[:8]}... (type: {type(SUPABASE_API_KEY)}, len: {len(SUPABASE_API_KEY)})")

# Initialize Supabase client
supabase = None
if SUPABASE_URL and SUPABASE_API_KEY:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
        print("✅ Successfully connected to Supabase!")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        traceback.print_exc()  # Add this line for full error details
        print("❌ Check if your SUPABASE_API_KEY is correct, not expired, and has the right permissions.")
else:
    print("❌ Missing Supabase credentials. Check your .env file.")

# Load database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Test database connection
if (DATABASE_URL):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ Successfully connected to the database!")
        conn.close()
    except Exception as e:
        print(f"❌ Connection failed to the database: {e}")
else:
    print("❌ DATABASE_URL not set. Skipping database connection test.")





# Load the Keras image classification model
image_model = tf.keras.models.load_model('model.h5')

# Class labels
CLASSES = [
    'Pigmented benign keratosis',
    'Melanoma',
    'Vascular lesion',
    'Actinic keratosis',
    'Squamous cell carcinoma',
    'Basal cell carcinoma',
    'Seborrheic keratosis',
    'Dermatofibroma',
    'Nevus'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image_path):
    """Process image for prediction"""
    img = Image.open(image_path)
    
    # Convert to RGB to ensure 3 channels (handles RGBA, grayscale, etc.)
    img = img.convert('RGB')
    
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # Verify shape has 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize using the same method as in training
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    
    return img_array







# --- Chatbot Model Loading and Response Function ---
logging.basicConfig(level=logging.DEBUG)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Ensure the model path is correct


model_path = os.path.join(project_root, "ggml-model-q4_0.gguf")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
else:
    logging.info(f"**************************Model file found at: {model_path}")

chatbot_model = None
chatbot_lock = threading.Lock()  # Always create a lock, even if model fails to load

def is_valid_gguf_file(filepath):
    # GGUF files start with magic bytes: b'GGUF'
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            return magic == b'GGUF'
    except Exception:
        return False

def load_chatbot_model():
    global chatbot_model
    try:
        if os.path.exists(model_path):
            # Check if file is a valid GGUF binary
            if not is_valid_gguf_file(model_path):
                logging.error(f"File at {model_path} is not a valid GGUF model file. "
                              "It may be an HTML error page or corrupted download. "
                              "Please re-download the model from the correct source.")
                with chatbot_lock:
                    chatbot_model = None
                return
            logging.info(f"Loading model from: {model_path}")
            temp_model = None
            try:
                temp_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type="llama",
                    context_length=256,
                    batch_size=1,
                    threads=1,
                    reset=True
                )
            except Exception as model_e:
                logging.error(f"Model loading failed: {model_e}")
                logging.error(traceback.format_exc())
            with chatbot_lock:
                chatbot_model = temp_model
            if temp_model is not None:
                chatbot_model = temp_model
                logging.info("Model loaded successfully ✅")
            else:
                logging.error("Model failed to load, chatbot_model is None")
        else:
            logging.error(f"Model file not found at: {model_path}")
            with chatbot_lock:
                chatbot_model = None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        with chatbot_lock:
            chatbot_model = None

# Load the chatbot model at startup
load_chatbot_model()

def generate_response(question):
    print("---------------------------",chatbot_model)	
    try:
        with chatbot_lock:
            model = chatbot_model
        if chatbot_model is None:
            fallback_responses = {
                "what is skin cancer": "Skin cancer is the abnormal growth of skin cells, most often developing on skin exposed to the sun. It's the most common form of cancer globally.",
                "types of skin cancer": "The three main types of skin cancer are:\n1. Basal cell carcinoma (BCC) - Most common and least dangerous\n2. Squamous cell carcinoma (SCC) - Second most common, can spread if untreated\n3. Melanoma - Less common but most dangerous, can spread rapidly",
                "symptoms of skin cancer": "Common symptoms include:\n- Changes in existing moles\n- New growths on the skin\n- Sores that don't heal\n- Unusual skin patches that are red, itchy, or scaly",
                "how to prevent skin cancer": "Key prevention methods:\n- Use sunscreen (SPF 30+)\n- Avoid tanning beds\n- Wear protective clothing\n- Seek shade during peak sun hours\n- Get regular skin checks"
            }
            for key, response in fallback_responses.items():
                if question.lower() in key or key in question.lower():
                    return response
            return "Sorry, the skin cancer model is not available. Please try asking about basic skin cancer information like types, symptoms, or prevention."
        prompt = f"Q:{question}\nA:"
        logging.info(f"Processing question: {question}")
        try:
            with chatbot_lock:
                output = chatbot_model(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    stop=["\n", "Q:", "Question:"]
                )
        except Exception as e:
            logging.error(f"Error during chatbot_model inference: {str(e)}")
            logging.error(traceback.format_exc())
            return "Sorry, there was an error generating a response. Please try again later."
        logging.info(f"Raw model output: {output}")
        answer = output.strip()
        if not answer:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        return answer
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error processing request: {str(e)}"












def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'userid' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/')
def home():
    return render_template("index.html")

# 🟢 Login Route
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        print(f"Received Email: {email}, Password: {password}")

        try:
            response = supabase.table('user').select('*').eq('email', email).single().execute()
            print(f"🔍 Supabase Response: {response}")
            User = response.data

            if User and 'password' in User and User['password'] == password:  # ⬅️ مقارنة مباشرة بدون hash
                session.clear()
                session.permanent = True
                session['userid'] = User['userid']
                session['email'] = User['email']
                session['role'] = User.get('role', 'user')

                flash('✅ Login successful!', 'success')
                return redirect(url_for('auth.dashboard' if User['role'] == 'admin' else 'auth.home'))  # ⬅️ return يمنع تنفيذ flash آخر

            flash('❌ Invalid email or password.', 'danger')

        except Exception as e:
            print(f"🔥 Exception: {e}") 
            flash('⚠ An error occurred during login. Please try again.', 'danger')

    return render_template('login.html')

from datetime import datetime

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # ✅ Get form data and strip extra spaces
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # ✅ Validate input fields
        if not email or not password or not confirm_password or not name:
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
            # ✅ Check if the email or username already exists
            existing_user = supabase.table('user').select('email', 'username').or_(
                f"email.eq.{email},username.eq.{name}"
            ).maybe_single().execute()
            
            print("Raw Response from DB:", existing_user)  # Debugging

            # Fix: Check if existing_user.data is a list or dict and handle accordingly
            existing_data = getattr(existing_user, 'data', None)
            if existing_data:
                # If Supabase returns a list, check each item
                if isinstance(existing_data, list):
                    for user in existing_data:
                        if user.get('email') == email:
                            flash('⚠ This email is already registered.', 'error')
                            return redirect(url_for('auth.signup'))
                        if user.get('username') == name:
                            flash('⚠ This username is already taken.', 'error')
                            return redirect(url_for('auth.signup'))
                # If Supabase returns a dict (single user)
                elif isinstance(existing_data, dict):
                    if existing_data.get('email') == email:
                        flash('⚠ This email is already registered.', 'error')
                        return redirect(url_for('auth.signup'))
                    if existing_data.get('username') == name:
                        flash('⚠ This username is already taken.', 'error')
                        return redirect(url_for('auth.signup'))

            # ✅ Insert new user into the database
            new_user = {
                "email": email,
            }

            insert_response = supabase.table('user').insert(new_user).execute()
            print("Insert response:", insert_response)  # Debugging

            # Fix: Log in the user immediately after signup
            user_id = None
            # Try to get the new user's id from the insert response
            if hasattr(insert_response, 'data') and insert_response.data:
                # If data is a list, get the first item
                if isinstance(insert_response.data, list):
                    user_id = insert_response.data[0].get('userid')
                elif isinstance(insert_response.data, dict):
                    user_id = insert_response.data.get('userid')
            # Fallback: fetch the user by email if id not found
            if not user_id:
                user_lookup = supabase.table('user').select('*').eq('email', email).single().execute()
                if hasattr(user_lookup, 'data') and user_lookup.data:
                    user_id = user_lookup.data.get('userid')

            # Set session if user_id found
            if user_id:
                session.clear()
                session.permanent = True
                session['userid'] = user_id
                session['email'] = email
                session['role'] = 'user'
                flash('🎉 Account created and logged in successfully!', 'success')
                return redirect(url_for('auth.home'))
            else:
                flash('⚠ Account created, but could not log in automatically. Please log in manually.', 'warning')
                return redirect(url_for('auth.login'))

        except Exception as e:
            flash('⚠ An error occurred during registration. Please try again.', 'danger')
            print("🔥 Signup error:", str(e))  # Debugging

    return render_template('signup.html')

@auth.route('/logout')
def logout():
    session.clear()
    # Remove session cookie explicitly and set session to be non-permanent
    resp = make_response(redirect(url_for('auth.login')))
    resp.set_cookie('session', '', expires=0, path='/')
    session.permanent = False
    flash('✅ Logged out successfully.', 'info')
    return resp

@auth.route('/dashboard')
@login_required
def dashboard():
    userid = session.get('userid')
    user_role = session.get('role', 'user')
    dashboard_data = {}
    blogs = []
    try:
        # Fetch all images and join with user and classification info for admin, or only user's images for normal user
        images = []
        if user_role == 'admin':
            images_resp = supabase.table('image').select('*').execute()
            images = images_resp.data if hasattr(images_resp, 'data') else []
            blogs_resp = supabase.table('blog').select('*').order('postdate', desc=True).execute()
            blogs = blogs_resp.data if hasattr(blogs_resp, 'data') else []
        else:
            images_resp = supabase.table('image').select('*').eq('userid', userid).execute()
            images = images_resp.data if hasattr(images_resp, 'data') else []

        # Fetch all users (for username lookup)
        user_map = {}
        users_resp = supabase.table('user').select('userid,username').execute()
        users_list = users_resp.data if hasattr(users_resp, 'data') else []
        for u in users_list:
            user_map[u['userid']] = u['username']

        # Attach username and classification info to each image
        for img in images:
            img['username'] = user_map.get(img['userid'], 'N/A')
            class_resp = supabase.table('classification').select('*').eq('imageid', img['imageid']).maybe_single().execute()
            classification = class_resp.data if hasattr(class_resp, 'data') else None
            img['classification'] = classification

            # --- Ensure image_path is a browser-accessible static path ---
            img_path = img.get('image_path', '')
            if img_path:
                filename = os.path.basename(img_path)
                # Save as /static/uploads/filename for browser access (like blog images)
                img['image_path'] = f'/static/uploads/{filename}'
            else:
                img['image_path'] = ''

        dashboard_data['images'] = images
    except Exception as e:
        print("Dashboard fetch error:", str(e))
        flash('Could not load dashboard data.', 'danger')
        dashboard_data['images'] = []
        
    users = []
    if user_role == 'admin':
        try:
            users_resp = supabase.table('user').select('userid,username,email,registrationdate').execute()
            users = users_resp.data if hasattr(users_resp, 'data') else []
        except Exception as e:
            print("Dashboard users fetch error:", str(e))
            flash('Could not load users data.', 'danger')
    return render_template(
        'dashboard.html',
        dashboard_data=dashboard_data,
        users=users,
        blogs=blogs,
        is_admin=(user_role == 'admin')
    )

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
@login_required
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
    try:
        blogs_resp = supabase.table('blog').select('*').order('postdate', desc=True).execute()
        blogs = blogs_resp.data if hasattr(blogs_resp, 'data') else []
    except Exception as e:
        blogs = []
        print("Blog fetch error:", str(e))
    # Pass all section fields to the template for rendering
    return render_template(
        'blog.html',
        blogs=blogs
    )

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
@login_required
def detect():
    return render_template('detect.html')



UPLOAD_FOLDER = 'website/static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@auth.route('/predict', methods=['POST'])
@login_required
def predict():
    # Support both AJAX and form POSTs
    file = request.files.get('file') or request.files.get('image')
    if not file or file.filename == '':
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'No file uploaded'}), 400
        flash('No file selected.', 'danger')
        return redirect(url_for('auth.detect'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            img_array = process_image(file_path)
            predictions = image_model.predict(img_array)
            predicted_class = CLASSES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]) * 100)
            print(f"Predicted class: {predicted_class}, Confidence: {confidence}")


            imageid = str(uuid.uuid4())
            upload_date = datetime.utcnow().isoformat()
            userid = session.get('userid')
            if not userid:
                if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': 'User not logged in'}), 400
                flash('User not logged in.', 'danger')
                return redirect(url_for('auth.login'))
            try:
                uuid.UUID(userid)
            except ValueError:
                if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': 'Invalid user ID format'}), 400
                flash('Invalid user ID format.', 'danger')
                return redirect(url_for('auth.login'))

            image_data = {
                "imageid": imageid,
                "userid": userid,
                "uploaddate": upload_date,
                "image_path": file_path,
            }
            image_result = supabase.table("image").insert(image_data).execute()
            if not getattr(image_result, "data", None):
                if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'error': f'Database error (image): {str(image_result.error)}. '
                                 f'Result: {str(image_result)}. '
                                 'Check table schema, NOT NULL constraints, and RLS policies.'
                    }), 500
                flash('Database error (image).', 'danger')
                return redirect(url_for('auth.detect'))

            classification_data = {
                "classificationid": str(uuid.uuid4()),
                "imageid": imageid,
                "result": predicted_class,
                "confidence": confidence,
                "classificationdate": upload_date
            }
            class_result = supabase.table("classification").insert(classification_data).execute()
            if not getattr(class_result, "data", None):
                if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'error': f'Database error (classification): {str(class_result.error)}. '
                                 f'Result: {str(class_result)}. '
                                 'Check table schema, NOT NULL constraints, and RLS policies.'
                    }), 500
                flash('Database error (classification).', 'danger')
                return redirect(url_for('auth.detect'))

            # If AJAX, return JSON
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'class': predicted_class,
                    'confidence': confidence,
                    'image_id': imageid,
                    'image_path': file_path
                })

            # If standard form POST, redirect to result page
            return redirect(url_for('auth.result', 
                                   predicted_class=predicted_class, 
                                   confidence=confidence, 
                                   image_path=filename))

        except Exception as e:
            print("Prediction error:", str(e))
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': f'Prediction error: {str(e)}'}), 500
            flash('Prediction error.', 'danger')
            return redirect(url_for('auth.detect'))

    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'error': 'Invalid file type'}), 400
    flash('Invalid file type.', 'danger')
    return redirect(url_for('auth.detect'))

@auth.route('/result')
@login_required
def result():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    image_path = request.args.get('image_path')
    # Debug: print image_path to verify value
    print("Result route image_path:", image_path)
    image_url = None
    if image_path:
        # Always use the filename only, not the full path
        image_filename = os.path.basename(image_path)
        # Always use the static path for browser access
        image_url = url_for('static', filename=f'uploads/{image_filename}')
        print("Result route image_filename:", image_filename)
        print("Result route image_url:", image_url)
    return render_template('result.html', 
                          predicted_class=predicted_class, 
                          confidence=confidence, 
                          image_url=image_url)

@auth.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



@auth.route('/delete_user/<userid>', methods=['POST'])
@login_required
def delete_user(userid):
    user_role = session.get('role', 'user')
    if user_role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        # Delete user from user table
        supabase.table('user').delete().eq('userid', userid).execute()
        # Optionally: delete user's images and classifications
        supabase.table('image').delete().eq('userid', userid).execute()
        # You may also want to delete classifications for those images
        # (not shown here for brevity)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@auth.route('/delete_image/<imageid>', methods=['POST'])
@login_required
def delete_image(imageid):
    try:
        # Delete image and its classification
        supabase.table('classification').delete().eq('imageid', imageid).execute()
        supabase.table('image').delete().eq('imageid', imageid).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@auth.route('/edit_password/<userid>', methods=['POST'])
@login_required
def edit_password(userid):
    user_role = session.get('role', 'user')
    if user_role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json()
    new_password = data.get('password')
    if not new_password or len(new_password) < 7:
        return jsonify({'success': False, 'message': 'Password too short'}), 400
    try:
        supabase.table('user').update({'password': new_password}).eq('userid', userid).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@auth.route('/add_blog', methods=['POST'])
@login_required
def add_blog():
    user_role = session.get('role', 'user')
    if user_role != 'admin':
        flash('Unauthorized.', 'danger')
        return redirect(url_for('auth.dashboard'))

    # Get text fields
    title = request.form.get('title', '').strip()
    overview = request.form.get('overview', '').strip()
    symptoms = request.form.get('symptoms', '').strip()
    causes = request.form.get('causes', '').strip()
    prevention = request.form.get('prevention', '').strip()
    treatment = request.form.get('treatment', '').strip()
    doctor = request.form.get('doctor', '').strip()

    # Get image files
    image_file = request.files.get('image_file')
    overview_file = request.files.get('overview_path')
    symptoms_file = request.files.get('symptoms_path')
    causes_file = request.files.get('causes_path')
    prevention_file = request.files.get('prevention_path')
    treatment_file = request.files.get('treatment_path')
    doctor_file = request.files.get('doctor_path')

    # Validate required fields
    if not title or not overview or not image_file or image_file.filename == '':
        flash('Title, overview, and main image are required.', 'danger')
        return redirect(url_for('auth.dashboard'))

    # Save images to static/uploads/blogs
    blog_upload_folder = os.path.join('website', 'static', 'uploads', 'blogs')
    os.makedirs(blog_upload_folder, exist_ok=True)

    def save_img(file):
        if file and file.filename:
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            path = os.path.join(blog_upload_folder, filename)
            file.save(path)
            return f'/static/uploads/blogs/{filename}'
        return None

    image_url = save_img(image_file)
    overview_path = save_img(overview_file)
    symptoms_path = save_img(symptoms_file)
    causes_path = save_img(causes_file)
    prevention_path = save_img(prevention_file)
    treatment_path = save_img(treatment_file)
    doctor_path = save_img(doctor_file)

    try:
        blogid = str(uuid.uuid4())
        userid = session.get('userid')
        supabase.table('blog').insert({
            'blogid': blogid,
            'userid': userid,
            'title': title,
            'postdate': datetime.utcnow().isoformat(),
            'image_url': image_url,
            'overview': overview,
            'overview_path': overview_path,
            'symptoms': symptoms,
            'symptoms_path': symptoms_path,
            'causes': causes,
            'causes_path': causes_path,
            'prevention': prevention,
            'prevention_path': prevention_path,
            'treatment': treatment,
            'treatment_path': treatment_path,
            'doctor': doctor,
            'doctor_path': doctor_path
        }).execute()
        flash('Blog post added!', 'success')
    except Exception as e:
        flash('Failed to add blog post.', 'danger')
        print("Blog add error:", str(e))
    return redirect(url_for('auth.dashboard'))

@auth.route('/edit_blog/<blog_id>', methods=['POST'])
@login_required
def edit_blog(blog_id):
    user_role = session.get('role', 'user')
    if user_role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json()
    title = data.get('title', '').strip()
    # Remove 'content' and update all section fields instead
    overview = data.get('overview', '').strip()
    symptoms = data.get('symptoms', '').strip()
    causes = data.get('causes', '').strip()
    prevention = data.get('prevention', '').strip()
    treatment = data.get('treatment', '').strip()
    doctor = data.get('doctor', '').strip()
    # Optionally handle image paths if you want to allow editing them

    if not title or not overview:
        return jsonify({'success': False, 'message': 'Title and overview are required'}), 400
    try:
        # Update all section fields
        update_fields = {
            'title': title,
            'overview': overview,
            'symptoms': symptoms,
            'causes': causes,
            'prevention': prevention,
            'treatment': treatment,
            'doctor': doctor
        }
        result = supabase.table('blog').update(update_fields).eq('blogid', blog_id).execute()
        if not getattr(result, "data", None):
            result2 = supabase.table('blog').update(update_fields).eq('id', blog_id).execute()
            if not getattr(result2, "data", None):
                return jsonify({'success': False, 'message': 'Blog post not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@auth.route('/delete_blog/<blog_id>', methods=['POST'])
@login_required
def delete_blog(blog_id):
    user_role = session.get('role', 'user')
    if user_role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    try:
        # Try both possible id fields for compatibility
        result = supabase.table('blog').delete().eq('blogid', blog_id).execute()
        if not getattr(result, "data", None):
            # Try with 'id' if 'blogid' didn't work
            result2 = supabase.table('blog').delete().eq('id', blog_id).execute()
            if not getattr(result2, "data", None):
                return jsonify({'success': False, 'message': 'Blog post not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    


@auth.route('/disease')
def disease():
    topic = request.args.get('topic', '').lower()
    topic_map = {
        'bbc': 'bbc.html',
        'scc': 'scc.html',
        'melanoma': 'melanoma.html',
        'bkl': 'bkl.html',
        'df': 'df.html',
        'vl': 'vl.html',
        'nv': 'nv.html',
        'ak': 'ak.html'
    }
    # If topic matches a static template, render it
    # (optional: keep this for legacy/static pages)
    if topic in topic_map:
        return render_template(topic_map[topic])
    # If topic matches a blog post, show its content in disease.html
    if topic:
        blog = None
        try:
            blogs_resp = supabase.table('blog').select('*').execute()
            blogs = blogs_resp.data if hasattr(blogs_resp, 'data') else []
            import re
            def slugify(s):
                return re.sub(r'[^a-z0-9]', '', s.lower())
            for b in blogs:
                if slugify(b.get('title', '')) == topic:
                    blog = b
                    break
        except Exception as e:
            blog = None
        return render_template('disease.html', blog=blog)
    # Fallback: show generic disease page
    return render_template('disease.html')



# --- Add Chatbot API Endpoint to Blueprint ---
@auth.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        response = generate_response(question)
        print("-------------------------", response)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'response': f"Server error: {str(e)}"}), 500

# def load_chatbot_model():
#     global chatbot_model
#     try:
#         if os.path.exists(model_path):
#             # Check if file is a valid GGUF binary
#             if not is_valid_gguf_file(model_path):
#                 logging.error(f"File at {model_path} is not a valid GGUF model file. "
#                               "It may be an HTML error page or corrupted download. "
#                               "Please re-download the model from the correct source.")
#                 with chatbot_lock:
#                     chatbot_model = None
#                 return
#             logging.info(f"Loading model from: {model_path}")
#             temp_model = None
#             try:
#                 temp_model = AutoModelForCausalLM.from_pretrained(
#                     model_path,
#                     model_type="llama",
#                     context_length=256,
#                     batch_size=1,
#                     threads=1,
#                     reset=True
#                 )
#             except Exception as model_e:
#                 logging.error(f"Model loading failed: {model_e}")
#                 logging.error(traceback.format_exc())
#             with chatbot_lock:
#                 chatbot_model = temp_model
#             if temp_model is not None:
#                 chatbot_model = temp_model
#                 logging.info("Model loaded successfully ✅")
#             else:
#                 logging.error("Model failed to load, chatbot_model is None")
#         else:
#             logging.error(f"Model file not found at: {model_path}")
#             with chatbot_lock:
#                 chatbot_model = None
#     except Exception as e:
#         logging.error(f"Error loading model: {str(e)}")
#         logging.error(traceback.format_exc())
#         with chatbot_lock:
#             chatbot_model = None

# # Load the chatbot model at startup
# load_chatbot_model()

# def generate_response(question):
#     print("---------------------------",chatbot_model)	
#     try:
#         with chatbot_lock:
#             model = chatbot_model
#         if chatbot_model is None:
#             fallback_responses = {
#                 "what is skin cancer": "Skin cancer is the abnormal growth of skin cells, most often developing on skin exposed to the sun. It's the most common form of cancer globally.",
#                 "types of skin cancer": "The three main types of skin cancer are:\n1. Basal cell carcinoma (BCC) - Most common and least dangerous\n2. Squamous cell carcinoma (SCC) - Second most common, can spread if untreated\n3. Melanoma - Less common but most dangerous, can spread rapidly",
#                 "symptoms of skin cancer": "Common symptoms include:\n- Changes in existing moles\n- New growths on the skin\n- Sores that don't heal\n- Unusual skin patches that are red, itchy, or scaly",
#                 "how to prevent skin cancer": "Key prevention methods:\n- Use sunscreen (SPF 30+)\n- Avoid tanning beds\n- Wear protective clothing\n- Seek shade during peak sun hours\n- Get regular skin checks"
#             }
#             for key, response in fallback_responses.items():
#                 if question.lower() in key or key in question.lower():
#                     return response
#             return "Sorry, the skin cancer model is not available. Please try asking about basic skin cancer information like types, symptoms, or prevention."
#         prompt = f"Q:{question}\nA:"
#         logging.info(f"Processing question: {question}")
#         try:
#             with chatbot_lock:
#                 output = chatbot_model(
#                     prompt,
#                     max_new_tokens=50,
#                     temperature=0.7,
#                     top_k=20,
#                     top_p=0.9,
#                     repetition_penalty=1.1,
#                     stop=["\n", "Q:", "Question:"]
#                 )
#         except Exception as e:
#             logging.error(f"Error during chatbot_model inference: {str(e)}")
#             logging.error(traceback.format_exc())
#             return "Sorry, there was an error generating a response. Please try again later."
#         logging.info(f"Raw model output: {output}")
#         answer = output.strip()
#         if not answer:
#             return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
#         return answer
#     except Exception as e:
#         logging.error(f"Error generating response: {str(e)}")
#         logging.error(traceback.format_exc())
#         return f"Error processing request: {str(e)}"