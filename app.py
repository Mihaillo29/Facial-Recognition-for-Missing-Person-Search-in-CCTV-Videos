from flask import Flask, render_template, redirect, url_for, request, flash, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, pickle, numpy as np, cv2, base64, uuid, math, hashlib, time
from datetime import datetime
from collections import deque
import concurrent.futures
from cachetools import LRUCache
import face_recognition
from flask_socketio import SocketIO, emit
from twilio.rest import Client

# Twilio configuration (update with your Twilio details or use environment variables)
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '#')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '#')
TWILIO_FROM_PHONE = os.environ.get('TWILIO_FROM_PHONE', "#")

app = Flask(__name__)
app.config['SECRET_KEY'] = '#'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///missing_persons.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MATCHES_FOLDER'] = 'static/matches'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}
app.config['FACE_DATA_PATH'] = 'face_data.pkl'
app.config['FACE_MATCH_THRESHOLD'] = 0.39

# Ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MATCHES_FOLDER'], exist_ok=True)


db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

def send_sms_alert(phone_number, context):
    """
    Sends an SMS alert using Twilio.
    Template:
      Dear {register_person_name},
      
      We are pleased to inform you that the missing person missing from {missing_from} and you were concerned about has been found. The person was located in a camera footage, and we have identified their whereabouts.
      
      Here are the details:
       - Name: {first_name} {last_name}
       - Date and Time of Sighting: {date_time}
      
      Thank you for your cooperation and concern in this matter.
    """
    # Format the phone number: if it's 10 digits, assume it's a local number and prepend "+91".
    phone_number = f"+91{phone_number}" if len(str(phone_number)) == 10 else f"+{phone_number}"
    
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message_body = (
        f"Dear {context['register_person_name']},\n\n"
        f"We are pleased to inform you that the missing person missing from {context['missing_from']} and you were concerned about has been found. "
        "The person was located in a camera footage, and we have identified their whereabouts.\n\n"
        "Here are the details:\n"
        f" - Name: {context['first_name']} {context['last_name']}\n"
        f" - Date and Time of Sighting: {context['date_time']}\n\n"
        "Thank you for your cooperation and concern in this matter.\n\n"
    )
    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_FROM_PHONE,
            to=phone_number
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100))
    phone_number = db.Column(db.String(20))  # New field for phone number
    password_hash = db.Column(db.String(200))
    missing_persons = db.relationship('MissingPerson', backref='reporter', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class MissingPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    last_seen_date = db.Column(db.DateTime)
    last_seen_location = db.Column(db.String(200))
    description = db.Column(db.Text)
    photo_path = db.Column(db.String(200))
    date_reported = db.Column(db.DateTime, default=datetime.utcnow)
    reporter_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# New model to track video frame matches (from first app)
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.String(100), nullable=False)
    missing_person_id = db.Column(db.Integer, db.ForeignKey('missing_person.id'), nullable=False)
    frame_path = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    frame_number = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    missing_person = db.relationship('MissingPerson')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------
# FUNCTIONS (Missing Persons Registry)
# ------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_face_data():
    if os.path.exists(app.config['FACE_DATA_PATH']):
        with open(app.config['FACE_DATA_PATH'], 'rb') as f:
            return pickle.load(f)
    return {'encodings': [], 'ids': []}

def save_face_data(face_data):
    with open(app.config['FACE_DATA_PATH'], 'wb') as f:
        pickle.dump(face_data, f)

def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog") # use cnn if u want "but u have to compile opencv with cuda"
    if not face_locations:
        return None
    face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
    return face_encoding

def register_missing_person_face(missing_person_id, image_path):
    face_encoding = encode_face(image_path)
    if face_encoding is None:
        return False
    face_data = load_face_data()
    face_data['encodings'].append(face_encoding)
    face_data['ids'].append(missing_person_id)
    save_face_data(face_data)
    return True

def find_missing_person_in_image(image_path, tolerance=None):
    if tolerance is None:
        tolerance = app.config.get('FACE_MATCH_THRESHOLD', 0.6)
    face_data = load_face_data()
    if not face_data['encodings']:
        return []
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog")
    if not face_locations:
        return []
    face_encodings = face_recognition.face_encodings(image, face_locations)
    matches = []
    known_encodings = np.array(face_data['encodings'])
    for i, face_encoding in enumerate(face_encodings):
        dists = np.linalg.norm(known_encodings - face_encoding, axis=1)
        similarities = np.exp(-dists)
        best_match_index = np.argmin(dists)
        if dists[best_match_index] < tolerance:
            person_id = face_data['ids'][best_match_index]
            confidence = similarities[best_match_index]
            matches.append((person_id, confidence, face_locations[i]))
    return matches

def find_missing_person_in_video(video_path, tolerance=None, search_id=None):
    if tolerance is None:
        tolerance = app.config.get('FACE_MATCH_THRESHOLD', 0.6)
    face_data = load_face_data()
    if not face_data['encodings']:
        return [],[]
    matches = {}
    known_encodings = np.array(face_data['encodings'])
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for i, face_encoding in enumerate(face_encodings):
                    dists = np.linalg.norm(known_encodings - face_encoding, axis=1)
                    similarities = np.exp(-dists)
                    best_match_index = np.argmin(dists)
                    if dists[best_match_index] < tolerance:
                        person_id = face_data['ids'][best_match_index]
                        confidence = similarities[best_match_index]
                        if person_id not in matches:
                            matches[person_id] = {'confidence': confidence, 'frames': []}
                        elif confidence > matches[person_id]['confidence']:
                            matches[person_id]['confidence'] = confidence
                        matches[person_id]['frames'].append((frame_count, face_locations[i], confidence))
                        if search_id:
                            top, right, bottom, left = face_locations[i]
                            marked_frame = frame.copy()
                            cv2.rectangle(marked_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            person = MissingPerson.query.get(person_id)
                            if person:
                                label = f"{person.name} ({confidence*100:.1f}%)"
                                cv2.putText(marked_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            frame_filename = f"{search_id}_{person_id}_{frame_count}.jpg"
                            frame_path = os.path.join(app.config['MATCHES_FOLDER'], frame_filename)
                            cv2.imwrite(frame_path, marked_frame)
                            from sqlalchemy.exc import SQLAlchemyError
                            try:
                                detection = DetectionResult(
                                    search_id=search_id,
                                    missing_person_id=person_id,
                                    frame_path=frame_filename,
                                    confidence=float(confidence),
                                    frame_number=frame_count
                                )
                                db.session.add(detection)
                            except SQLAlchemyError as e:
                                print(f"DB error: {str(e)}")
                            saved_frames.append((person_id, frame_filename, confidence, frame_count))
        frame_count += 1
    video.release()
    if search_id:
        db.session.commit()
    return [(person_id, matches[person_id]['confidence']) for person_id in matches], saved_frames

face_distance_threshold = 0.6  # typical threshold
detection_threshold = math.exp(-face_distance_threshold)  # similarity threshold (~0.55)
embedding_cache = LRUCache(maxsize=500)
frame_similarity_buffers = {}
processing_videos = {}
reference_embeddings = {}

# Face detection model (OpenCV DNN)
prototxt_path = os.path.join("models", "deploy.prototxt.txt")
weights_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
dnn_conf_threshold = 0.12

def detect_faces(frame, conf_threshold=dnn_conf_threshold):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

def get_face_embedding(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_hash = hashlib.md5(rgb_face.tobytes()).hexdigest()
    if img_hash in embedding_cache:
        print("[DEBUG] Using cached embedding.")
        return embedding_cache[img_hash]
    face_locations = [(0, rgb_face.shape[1], rgb_face.shape[0], 0)]
    encodings = face_recognition.face_encodings(rgb_face, face_locations)
    if encodings:
        embedding_cache[img_hash] = encodings[0]
        print("[DEBUG] Computed and cached new embedding.")
        return encodings[0]
    raise ValueError("No face encoding found.")

def compute_similarity(emb1, emb2, threshold=face_distance_threshold):
    dist = np.linalg.norm(emb1 - emb2)
    similarity = np.exp(-dist)
    detected = dist < threshold
    print(f"[DEBUG] CPU similarity computed: distance={dist:.4f}, similarity={similarity:.4f}")
    return similarity, detected, dist

def process_single_face(face_data, video_id):
    face_img, (x, y, w, h) = face_data
    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
        return None
    try:
        face_embedding = get_face_embedding(face_img)
        results = []
        for ref_id, ref_data in reference_embeddings.items():
            ref_embedding = ref_data['embedding']
            ref_name = ref_data['name']
            similarity, detected, dist = compute_similarity(ref_embedding, face_embedding, threshold=face_distance_threshold)
            results.append({
                'ref_id': ref_id,
                'name': ref_name,
                'similarity': similarity,
                'detected': detected,
                'distance': dist,
                'position': (x, y, w, h)
            })
        if results:
            return max(results, key=lambda x: x['similarity'])
        return None
    except Exception as e:
        print(f"Error processing face: {str(e)}")
        return None

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes_array = np.array(boxes)
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 0] + boxes_array[:, 2]
    y2 = boxes_array[:, 1] + boxes_array[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)[::-1]
    pick = []
    while len(idxs) > 0:
        current = idxs[0]
        pick.append(current)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (area[current] + area[rest] - inter)
        idxs = idxs[np.where(iou <= overlapThresh)[0] + 1]
    filtered_boxes = boxes_array[pick]
    return [tuple(box) for box in filtered_boxes]

def process_frame(frame, video_id):
    if not reference_embeddings:
        return frame, False
    frame_copy = frame.copy()
    faces = detect_faces(frame)
    faces = non_max_suppression(faces, overlapThresh=0.3)
    face_data = [(frame[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]
    face_results = []
    for face in face_data:
        result = process_single_face(face, video_id)
        if result is not None:
            face_results.append(result)
    person_detected = False
    max_similarity = 0
    for result in face_results:
        x, y, w, h = result['position']
        similarity = result['similarity']
        max_similarity = max(max_similarity, similarity)
        color = (0, 255, 0) if result['detected'] else (0, 0, 255)
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
        label_text = f"{similarity:.2f}"
        if result['detected']:
            label_text = f"{result['name']}: {similarity:.2f}"
            person_detected = True
        cv2.putText(frame_copy, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Video {video_id}: Face at {(x, y, w, h)} => similarity {similarity:.2f} (distance: {result['distance']:.2f})")
    if video_id not in frame_similarity_buffers:
        frame_similarity_buffers[video_id] = deque(maxlen=5)
    frame_similarity_buffers[video_id].append(max_similarity)
    avg_similarity = np.mean(frame_similarity_buffers[video_id]) if frame_similarity_buffers[video_id] else 0
    person_detected = avg_similarity > detection_threshold
    return frame_copy, person_detected

executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def process_video(video_path, video_id):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 2) if fps > 0 else 5
        frame_count = 0
        processing_videos[video_id]['status'] = 'processing'
        frame_similarity_buffers[video_id] = deque(maxlen=5)
        futures = []
        while cap.isOpened() and processing_videos.get(video_id, {}).get('status') == 'processing':
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                future = executor.submit(process_frame, frame.copy(), video_id)
                futures.append(future)
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    try:
                        processed_frame, person_detected = future.result()
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        socketio.emit('frame_update', {
                            'video_id': video_id,
                            'frame': jpg_as_text,
                            'detected': bool(person_detected)
                        })
                    except Exception as e:
                        print(f"Error in background task for video {video_id}: {str(e)}")
                    futures.remove(future)
                socketio.sleep(0.05)
        concurrent.futures.wait(futures)
        cap.release()
        processing_videos[video_id]['status'] = 'completed'
        socketio.emit('video_completed', {'video_id': video_id})
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        processing_videos[video_id]['status'] = 'error'
        socketio.emit('video_error', {'video_id': video_id, 'error': str(e)})

def start_video_processing(video_ids):
    for video_id in video_ids:
        if video_id in processing_videos and processing_videos[video_id]['status'] == 'pending':
            process_video(processing_videos[video_id]['path'], video_id)

# ------------------------
# ROUTES (Missing Persons Registry)
# ------------------------

@app.route('/')
def home():
    return render_template('index.html', section='home')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        phone_number = request.form.get('phone_number')  # Get phone number from form
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, phone_number=phone_number)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('index.html', section='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('index.html', section='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', section='dashboard')

@app.route('/register_missing', methods=['GET', 'POST'])
@login_required
def register_missing():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        last_seen_date = datetime.strptime(request.form.get('last_seen_date'), '%Y-%m-%d')
        last_seen_location = request.form.get('last_seen_location')
        description = request.form.get('description')
        if 'photo' not in request.files:
            flash('No photo uploaded')
            return redirect(url_for('register_missing'))
        photo = request.files['photo']
        if photo.filename == '':
            flash('No photo selected')
            return redirect(url_for('register_missing'))
        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_{filename}"
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(photo_path)
            missing_person = MissingPerson(
                name=name,
                age=age,
                gender=gender,
                last_seen_date=last_seen_date,
                last_seen_location=last_seen_location,
                description=description,
                photo_path=filename,
                reporter_id=current_user.id
            )
            db.session.add(missing_person)
            db.session.commit()
            if register_missing_person_face(missing_person.id, photo_path):
                flash('Missing person registered successfully with face recognition')
            else:
                flash('Missing person registered but no face detected in the photo')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid file type. Please upload a jpg, jpeg, or png file.')
            return redirect(url_for('register_missing'))
    return render_template('index.html', section='register_missing')

@app.route('/find_missing', methods=['GET', 'POST'])
@login_required
def find_missing():
    results = []
    frame_detections = []
    if request.method == 'POST':
        threshold_input = request.form.get('threshold')
        try:
            tolerance = float(threshold_input) if threshold_input else app.config.get('FACE_MATCH_THRESHOLD', 0.6)
        except ValueError:
            tolerance = app.config.get('FACE_MATCH_THRESHOLD', 0.6)
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('find_missing'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('find_missing'))
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_type = request.form.get('file_type')
            search_id = f"search_{timestamp}"
            if file_type == 'image':
                matches = find_missing_person_in_image(file_path, tolerance=tolerance)
                if matches:
                    image = cv2.imread(file_path)
                    for person_id, confidence, face_location in matches:
                        top, right, bottom, left = face_location
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                        person = MissingPerson.query.get(person_id)
                        if person:
                            label = f"{person.name} ({confidence*100:.1f}%)"
                            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    annotated_filename = f"{search_id}_annotated.jpg"
                    annotated_path = os.path.join(app.config['MATCHES_FOLDER'], annotated_filename)
                    cv2.imwrite(annotated_path, image)
                    for person_id, confidence, _ in matches:
                        missing_person = MissingPerson.query.get(person_id)
                        if missing_person:
                            results.append({
                                'person': missing_person,
                                'confidence': round(confidence * 100, 2),
                                'annotated_image': annotated_filename
                            })
                # If there are detection results, send SMS alerts to each unique reporter.
                if results:
                    sent_sms = set()
                    for result in results:
                        missing_person = result['person']
                        reporter = missing_person.reporter
                        if reporter and reporter.phone_number and reporter.id not in sent_sms:
                            context = {
                                'register_person_name': reporter.username,
                                'missing_from': missing_person.last_seen_location,
                                # Here we assume missing_person.name holds the full name.
                                'first_name': missing_person.name,  
                                'last_name': '',  # No separate last name provided
                                'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            send_sms_alert(reporter.phone_number, context)
                            sent_sms.add(reporter.id)
            else:
                # Video branch remains similar; you can add similar SMS logic here if needed.
                matches, saved_frames = find_missing_person_in_video(
                    file_path, tolerance=tolerance, search_id=search_id
                )
                for person_id, confidence in matches:
                    missing_person = MissingPerson.query.get(person_id)
                    if missing_person:
                        results.append({
                            'person': missing_person,
                            'confidence': round(confidence * 100, 2)
                        })
                frame_detections = DetectionResult.query.filter_by(search_id=search_id).order_by(
                    DetectionResult.missing_person_id,
                    DetectionResult.confidence.desc()
                ).all()
                # Optionally send SMS for video detections similarly.
            if not results:
                flash('No matches found')
    return render_template('index.html', section='find_missing', 
                           results=results, frame_detections=frame_detections,
                           is_video=request.method == 'POST' and request.form.get('file_type') == 'video')


@app.route('/my_reports')
@login_required
def my_reports():
    reports = MissingPerson.query.filter_by(reporter_id=current_user.id).all()
    return render_template('index.html', section='my_reports', reports=reports)

@app.route('/view_missing/<int:id>')
def view_missing(id):
    missing_person = MissingPerson.query.get_or_404(id)
    return render_template('index.html', section='view_missing', person=missing_person)

# @app.route('/realtime')
# @login_required
# def realtime():
#     return redirect(url_for('realtime'))

@app.route('/video_feed')
@login_required
def video_feed():
    def gen():
        with app.app_context():
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    face_data = load_face_data()
                    known_encodings = np.array(face_data['encodings'])
                    for i, face_encoding in enumerate(face_encodings):
                        if known_encodings.size > 0:
                            dists = np.linalg.norm(known_encodings - face_encoding, axis=1)
                            tolerance = app.config.get('FACE_MATCH_THRESHOLD', 0.6)
                            best_match_index = np.argmin(dists)
                            if dists[best_match_index] < tolerance:
                                person_id = face_data['ids'][best_match_index]
                                person = MissingPerson.query.get(person_id)
                                if person:
                                    label = f"{person.name}"
                                    top, right, bottom, left = face_locations[i]
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                    cv2.putText(frame, label, (left, top - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            cap.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# New route for CCTV realtime section (from second app)
@app.route('/cctv_realtime')
@login_required
def cctv_realtime():
    return render_template('index.html', section='cctv_realtime')

@app.route('/realtime')
@login_required
def realtime():
    return render_template('index.html', section='realtime')

# CCTV realtime AJAX routes (from second app)
@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    if 'reference_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    file = request.files['reference_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    reference_name = request.form.get('reference_name', 'Unknown')
    try:
        reference_id = str(uuid.uuid4())
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], f"reference_{reference_id}.jpg")
        file.save(reference_path)
        ref_img = cv2.imread(reference_path)
        faces = detect_faces(ref_img)
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected in reference image'})
        x, y, w, h = faces[0]
        face_img = ref_img[y:y+h, x:x+w]
        reference_embedding = get_face_embedding(face_img)
        reference_embeddings[reference_id] = {
            'embedding': reference_embedding,
            'name': reference_name,
            'path': reference_path
        }
        print(f"[DEBUG] Reference image processed for {reference_name} with ID {reference_id}")
        return jsonify({'success': True, 'message': f'Reference image for {reference_name} processed successfully',
                        'reference_id': reference_id, 'image_path': reference_path, 'reference_name': reference_name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_reference', methods=['POST'])
def delete_reference():
    data = request.get_json()
    reference_id = data.get('reference_id')
    if not reference_id or reference_id not in reference_embeddings:
        return jsonify({'success': False, 'error': 'Reference ID not found'})
    ref_data = reference_embeddings.pop(reference_id)
    try:
        if os.path.exists(ref_data['path']):
            os.remove(ref_data['path'])
    except Exception as e:
        print(f"Error deleting reference file: {str(e)}")
    print(f"[DEBUG] Reference for {ref_data['name']} deleted.")
    return jsonify({'success': True, 'message': f'Reference for {ref_data["name"]} deleted successfully'})

@app.route('/upload_videos', methods=['POST'])
def upload_videos():
    if 'videos' not in request.files:
        return jsonify({'success': False, 'error': 'No videos provided'})
    video_files = request.files.getlist('videos')
    if not video_files or video_files[0].filename == '':
        return jsonify({'success': False, 'error': 'No videos selected'})
    video_ids = []
    try:
        for video_file in video_files:
            video_id = str(uuid.uuid4())
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
            video_file.save(video_path)
            processing_videos[video_id] = {'path': video_path, 'status': 'pending', 'filename': video_file.filename}
            video_ids.append({'id': video_id, 'filename': video_file.filename})
            print(f"[DEBUG] Video uploaded: {video_file.filename} with ID {video_id}")
        return jsonify({'success': True, 'message': f'{len(video_ids)} videos uploaded successfully', 'videos': video_ids})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    data = request.get_json()
    video_ids = data.get('video_ids', [])
    if not video_ids:
        return jsonify({'success': False, 'error': 'No video IDs provided'})
    if not reference_embeddings:
        return jsonify({'success': False, 'error': 'No reference images processed yet'})
    
    print(f"[DEBUG] Starting processing for video IDs: {video_ids}")
    
    if video_ids and isinstance(video_ids[0], dict):
        video_id_list = [vid['id'] for vid in video_ids]
    else:
        video_id_list = video_ids

    socketio.start_background_task(start_video_processing, video_id_list)
    
    return jsonify({'success': True, 'message': f'Started processing {len(video_ids)} videos'})


@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    data = request.get_json()
    video_id = data.get('video_id')
    if video_id in processing_videos:
        processing_videos[video_id]['status'] = 'stopped'
        print(f"[DEBUG] Processing stopped for video ID: {video_id}")
        return jsonify({'success': True, 'message': 'Processing stopped'})
    return jsonify({'success': False, 'error': 'Video not found'})

@app.route('/get_references', methods=['GET'])
def get_references():
    refs = []
    for ref_id, ref_data in reference_embeddings.items():
        refs.append({'id': ref_id, 'name': ref_data['name'], 'path': ref_data['path']})
    return jsonify({'success': True, 'references': refs})

@socketio.on('connect')
def handle_connect():
    print('[DEBUG] Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('[DEBUG] Client disconnected')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
