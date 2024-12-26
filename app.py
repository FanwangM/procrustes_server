from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
from procrustes import orthogonal, rotational, permutation
from celery_config import celery
import tempfile
from datetime import datetime
import uuid
import threading
import shutil

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
file_lock = threading.Lock()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'npz', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_unique_upload_dir():
    """Create a unique directory for each upload session."""
    unique_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def clean_upload_dir(directory):
    """Safely clean up upload directory."""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")

def load_data(filepath):
    """Load data from various file formats."""
    try:
        ext = filepath.rsplit('.', 1)[1].lower()
        if ext == 'npz':
            with np.load(filepath) as data:
                return data['arr_0'] if 'arr_0' in data else next(iter(data.values()))
        elif ext == 'txt':
            return np.loadtxt(filepath)
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
            return df.to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading file {filepath}: {str(e)}")

def save_data(data, format_type):
    """Save data in the specified format."""
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, f'result.{format_type}')

    if format_type == 'npz':
        np.savez(filename, result=data)
    elif format_type == 'txt':
        np.savetxt(filename, data)
    elif format_type in ['xlsx', 'xls']:
        pd.DataFrame(data).to_excel(filename, index=False)

    return filename

def get_default_parameters(algorithm):
    """Get default parameters for each Procrustes algorithm."""
    if algorithm == 'orthogonal':
        return {
            'translate': True,
            'scale': True
        }
    elif algorithm == 'rotational':
        return {
            'translate': True
        }
    elif algorithm == 'permutation':
        return {
            "pad": True,
            "translate": False,
            "scale": False,
            "unpad_col": False,
            "unpad_row": False,
            "check_finite": True,
            "weight": None
        }
    return {}

@app.route('/get_default_params/<algorithm>')
def get_default_params(algorithm):
    """API endpoint to get default parameters for an algorithm."""
    return jsonify(get_default_parameters(algorithm))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_default_params/<algorithm>')
def default_params(algorithm):
    return jsonify(get_default_params(algorithm))

@celery.task(bind=True)
def process_matrices(self, algorithm, params, matrix1_data, matrix2_data):
    # Convert lists back to numpy arrays
    matrix1 = np.asarray(matrix1_data, dtype=float)
    matrix2 = np.asarray(matrix2_data, dtype=float)
    if matrix1.size == 0 or matrix2.size == 0:
        raise ValueError("Empty matrix received")
    # Process based on algorithm
    if algorithm == 'orthogonal':
        result = orthogonal(matrix1, matrix2, **params)
    elif algorithm == 'rotational':
        result = rotational(matrix1, matrix2, **params)
    elif algorithm == 'permutation':
        result = permutation(matrix1, matrix2, **params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
           # Convert numpy arrays to lists for JSON serialization
    try:
        if hasattr(result, 't'):
            transformation = result.t
        elif hasattr(result, 't1'):
            transformation = result.t1
        else:
            print("Warning: No transformation matrix found in result")
            transformation = np.eye(array1.shape[1])
        if hasattr(result, 'new_array'):
            new_array = result.new_array
        elif hasattr(result, 'array_transformed'):
            new_array = result.array_transformed
        else:
            print("Warning: No transformed array found in result")
            new_array = array2
        response_data = {
            'error': float(result.error),
            'transformation': transformation.tolist(),
            'new_array': new_array.tolist()
        }
        print("Response data prepared:", response_data)  # Debug log
        return jsonify(response_data)

    except Exception as e:
        return {'error': f"Processing error: {str(e)}"}

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received upload request")

    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    algorithm = request.form.get('algorithm', 'orthogonal')

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected files'}), 400

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    # Create a unique directory for this upload
    upload_dir = get_unique_upload_dir()

    try:
        # Parse parameters
        try:
            parameters = json.loads(request.form.get('parameters', '{}'))
        except json.JSONDecodeError:
            parameters = get_default_parameters(algorithm)

        # Save files with unique names
        file1_path = os.path.join(upload_dir, secure_filename(str(uuid.uuid4()) + '_' + file1.filename))
        file2_path = os.path.join(upload_dir, secure_filename(str(uuid.uuid4()) + '_' + file2.filename))

        with file_lock:
            file1.save(file1_path)
            file2.save(file2_path)

        # Load data
        array1 = load_data(file1_path)
        array2 = load_data(file2_path)

        print(f"Arrays loaded - shapes: {array1.shape}, {array2.shape}")

        # Perform Procrustes analysis
        if algorithm == 'orthogonal':
            result = orthogonal(array1, array2, **parameters)
        elif algorithm == 'rotational':
            result = rotational(array1, array2, **parameters)
        elif algorithm == 'permutation':
            result = permutation(array1, array2, **parameters)
        else:
            raise ValueError('Invalid algorithm')

        # Extract results
        if hasattr(result, 't'):
            transformation = result.t
        elif hasattr(result, 't1'):
            transformation = result.t1
        else:
            transformation = np.eye(array1.shape[1])

        if hasattr(result, 'new_array'):
            new_array = result.new_array
        elif hasattr(result, 'array_transformed'):
            new_array = result.array_transformed
        else:
            new_array = array2

        response_data = {
            'error': float(result.error),
            'transformation': transformation.tolist(),
            'new_array': new_array.tolist()
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the unique upload directory
        clean_upload_dir(upload_dir)

@app.route('/status/<task_id>')
def task_status(task_id):
    task = process_matrices.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is pending...'
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    else:
        response = {
            'state': task.state,
            'status': task.info
        }
    return jsonify(response)

@app.route('/download', methods=['POST'])
def download():
    try:
        data = json.loads(request.form['data'])
        format_type = request.form['format']

        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'procrustes_result_{timestamp}'

        if format_type == 'npz':
            filepath = os.path.join(temp_dir, f'{filename}.npz')
            np.savez(filepath, np.array(data))
        elif format_type == 'xlsx':
            filepath = os.path.join(temp_dir, f'{filename}.xlsx')
            pd.DataFrame(data).to_excel(filepath, index=False)
        else:  # txt
            filepath = os.path.join(temp_dir, f'{filename}.txt')
            np.savetxt(filepath, np.array(data))

        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
