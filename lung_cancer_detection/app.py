import os
import secrets
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, url_for, flash, redirect, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Load the trained model
MODEL_PATH = 'models/lung_cancer_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names (must match your training order)
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
CLASS_COLORS = {
    'Benign': 'success',
    'Malignant': 'danger',
    'Normal': 'info'
}
CLASS_DESCRIPTIONS = {
    'Benign': 'Non-cancerous growth that does not spread to other parts of the body. Regular monitoring recommended.',
    'Malignant': 'Cancerous growth that can invade nearby tissues and spread. Immediate medical consultation required.',
    'Normal': 'No signs of abnormal growth or cancer. Regular health check-ups recommended.'
}

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date_predicted = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Custom decorator for admin routes (if needed)
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.username != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    
    return render_template('login.html', title='Login')

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
                          .order_by(Prediction.date_predicted.desc())\
                          .limit(5).all()
    
    # Get prediction counts by type
    benign_count = Prediction.query.filter_by(user_id=current_user.id, prediction='Benign').count()
    malignant_count = Prediction.query.filter_by(user_id=current_user.id, prediction='Malignant').count()
    normal_count = Prediction.query.filter_by(user_id=current_user.id, prediction='Normal').count()
    
    return render_template('dashboard.html', 
                         title='Dashboard',
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions,
                         benign_count=benign_count,
                         malignant_count=malignant_count,
                         normal_count=normal_count)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = make_prediction(filepath)
            
            if result:
                prediction, confidence = result
                
                # Save to database
                pred = Prediction(
                    image_filename=filename,
                    prediction=prediction,
                    confidence=confidence,
                    user_id=current_user.id
                )
                db.session.add(pred)
                db.session.commit()
                
                flash(f'Prediction completed! Result: {prediction}', 'success')
                
                return render_template('result.html',
                                     title='Prediction Result',
                                     prediction=prediction,
                                     confidence=confidence,
                                     image_path=url_for('static', filename=f'uploads/{filename}'),
                                     description=CLASS_DESCRIPTIONS[prediction],
                                     color=CLASS_COLORS[prediction])
            else:
                flash('Error making prediction. Please try again.', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG).', 'danger')
            return redirect(request.url)
    
    return render_template('predict.html', title='Predict')

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.date_predicted.desc())\
                    .paginate(page=page, per_page=10)
    return render_template('history.html', title='History', predictions=predictions)

@app.route('/delete_prediction/<int:pred_id>')
@login_required
def delete_prediction(pred_id):
    prediction = Prediction.query.get_or_404(pred_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id:
        flash('You do not have permission to delete this prediction.', 'danger')
        return redirect(url_for('history'))
    
    # Delete image file
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], prediction.image_filename))
    except:
        pass
    
    db.session.delete(prediction)
    db.session.commit()
    
    flash('Prediction deleted successfully!', 'success')
    return redirect(url_for('history'))

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save temporary file
        filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = make_prediction(filepath)
        
        # Clean up temp file
        os.remove(filepath)
        
        if result:
            prediction, confidence = result
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'description': CLASS_DESCRIPTIONS[prediction]
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_prediction(image_path):
    """Make prediction using the loaded model"""
    if model is None:
        flash('Model not loaded. Please contact administrator.', 'danger')
        return None
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/api/prediction/<int:pred_id>')
@login_required
def get_prediction_details(pred_id):
    """API endpoint to get prediction details"""
    prediction = Prediction.query.get_or_404(pred_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id and current_user.username != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    # Get user details
    user = User.query.get(prediction.user_id)
    
    # Prepare probabilities (you might want to store these in your database)
    # For now, we'll create dummy probabilities based on the prediction
    probabilities = {
        'Benign': 0.0,
        'Malignant': 0.0,
        'Normal': 0.0
    }
    probabilities[prediction.prediction] = prediction.confidence
    
    # Distribute remaining probability among other classes
    remaining = 1.0 - prediction.confidence
    other_classes = [c for c in probabilities.keys() if c != prediction.prediction]
    if other_classes:
        per_class = remaining / len(other_classes)
        for c in other_classes:
            probabilities[c] = per_class
    
    return jsonify({
        'success': True,
        'prediction': {
            'id': prediction.id,
            'image_filename': prediction.image_filename,
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'date_predicted': prediction.date_predicted.isoformat(),
            'username': user.username,
            'description': CLASS_DESCRIPTIONS[prediction.prediction],
            'probabilities': probabilities,
            'model_version': '1.0',
            'image_size': '224x224'
        }
    })

@app.route('/prediction/share/<int:pred_id>')
@login_required
def share_prediction(pred_id):
    """Public share page for a prediction"""
    prediction = Prediction.query.get_or_404(pred_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id and current_user.username != 'admin':
        flash('You do not have permission to share this prediction.', 'danger')
        return redirect(url_for('history'))
    
    return render_template('share.html',
                         prediction=prediction,
                         description=CLASS_DESCRIPTIONS[prediction.prediction],
                         color=CLASS_COLORS[prediction.prediction])

@app.route('/api/prediction/<int:pred_id>/export')
@login_required
def export_prediction(pred_id):
    """Export prediction as PDF/JSON"""
    prediction = Prediction.query.get_or_404(pred_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id and current_user.username != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    # For now, return JSON
    return jsonify({
        'id': prediction.id,
        'prediction': prediction.prediction,
        'confidence': prediction.confidence,
        'date': prediction.date_predicted.isoformat(),
        'image': prediction.image_filename
    })

@app.context_processor
def utility_processor():
    return {'now': datetime.now}

# Create database tables
with app.app_context():
    db.create_all()
    
    # Create admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            password=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created!")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)