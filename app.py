from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import fitz  # PyMuPDF for PDF handling
from werkzeug.utils import secure_filename
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json  # Import json module for handling JSON files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HISTORY_FILE'] = 'history.json'  # Define the path for the history file
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size of 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load tokenizer and label encoder
tokenizer = joblib.load('tokenizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the TensorFlow model
try:
    model = load_model('model.h5')
    print("Đã tải mô hình TensorFlow thành công.")
except Exception as e:
    raise Exception(f"Đã xảy ra lỗi khi tải mô hình TensorFlow: {e}")

# Initialize history from the history file or create an empty list
def load_history():
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Lỗi khi đọc file history.json. Khởi tạo lịch sử trống.")
            return []
    else:
        return []

def save_history():
    try:
        with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Đã xảy ra lỗi khi lưu lịch sử: {e}")

history = load_history()  # Load existing history

# Define prediction function
def predict(text):
    # Preprocess text
    sequences = tokenizer.texts_to_sequences([text])
    X = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    # Perform prediction
    prediction = model.predict(X)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)
    # Decode label
    label = label_encoder.inverse_transform(predicted_class)
    return label[0], confidence * 100  # Returns label and confidence percentage

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html template

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Extract text from PDF
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
        except Exception as e:
            return f"Đã xảy ra lỗi khi đọc file PDF: {e}"
        
        # Make prediction
        try:
            predicted_label, confidence_percentage = predict(text)
            # Add to history
            history_entry = {
                'label': predicted_label,
                'confidence': f"{confidence_percentage:.2f}%",
                'image_url': url_for('static', filename='uploads/' + filename)
            }
            history.append(history_entry)
            save_history()  # Save the updated history to the file
            return render_template(
                'result.html',
                label=predicted_label,
                percentage=confidence_percentage,
                image_url=history_entry['image_url']
            )
        except Exception as e:
            return f"Đã xảy ra lỗi khi dự đoán: {e}"
    else:
        return 'Invalid file format. Please upload a PDF file.'

@app.route('/history')
def view_history():
    return render_template('ai.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)