#!/usr/bin/env python3
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import subprocess
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    # If there's a matches.csv in repo root, show that as available
    default_exists = os.path.exists(os.path.join(BASE_DIR, 'matches.csv'))
    return render_template('index.html', default_exists=default_exists)


@app.route('/run', methods=['POST'])
def run_pipeline():
    use_default = request.form.get('use_default') == 'on'
    uploaded_path = None

    if use_default and os.path.exists(os.path.join(BASE_DIR, 'matches.csv')):
        uploaded_path = os.path.join(BASE_DIR, 'matches.csv')
    else:
        # handle file upload
        if 'file' not in request.files:
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            file.save(uploaded_path)
        else:
            return render_template('index.html', error='Invalid file. Please upload a CSV file.')

    # Run the pipeline script using the same Python interpreter
    cmd = [sys.executable, os.path.join(BASE_DIR, 'ipl_pipeline.py'), uploaded_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        output = 'Pipeline execution timed out.'

    return render_template('results.html', output=output)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
