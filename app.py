# import json
import os
import shutil
import tempfile
import threading
import uuid
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
# originally use jsonify from flask, but it doesn't support numpy array
from flask import Flask,render_template, request, send_file, Response
from procrustes import orthogonal, permutation, rotational
from werkzeug.utils import secure_filename
from celery_config import celery
import orjson
from flask_status import FlaskStatus
import markdown

app = Flask(__name__)
app_status = FlaskStatus(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"
file_lock = threading.Lock()

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "npz", "xlsx", "xls"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_unique_upload_dir():
    """Create a unique directory for each upload session."""
    unique_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(uuid.uuid4()))
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
        ext = filepath.rsplit(".", 1)[1].lower()
        if ext == "npz":
            with np.load(filepath) as data:
                return data["arr_0"] if "arr_0" in data else next(iter(data.values()))
        elif ext == "txt":
            return np.loadtxt(filepath)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(filepath)
            return df.to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading file {filepath}: {str(e)}")


def save_data(data, format_type):
    """Save data in the specified format."""
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, f"result.{format_type}")

    if format_type == "npz":
        np.savez(filename, result=data)
    elif format_type == "txt":
        np.savetxt(filename, data)
    elif format_type in ["xlsx", "xls"]:
        pd.DataFrame(data).to_excel(filename, index=False)

    return filename


def get_default_parameters(algorithm):
    """Get default parameters for each Procrustes algorithm."""
    if algorithm == "orthogonal":
        return {"translate": True, "scale": True}
    elif algorithm == "rotational":
        return {"translate": True}
    elif algorithm == "permutation":
        return {
            "pad": True,
            "translate": False,
            "scale": False,
            "unpad_col": False,
            "unpad_row": False,
            "check_finite": True,
            "weight": None,
        }
    return {}


def create_json_response(data, status=200):
    """Create a JSON response using orjson for better numpy array handling"""
    return Response(
        orjson.dumps(
            data,
            option=orjson.OPT_SERIALIZE_NUMPY,
            default=str
        ),
        status=status,
        mimetype='application/json'
    )


def read_markdown_file(filename):
    """Read and convert markdown file to HTML."""
    filepath = os.path.join(os.path.dirname(__file__), 'md_files', filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            # Pre-process math blocks to protect them
            content = content.replace('\\\\', '\\\\\\\\')  # Escape backslashes in math

            # Convert markdown to HTML with math and table support
            md = markdown.Markdown(extensions=[
                'tables',
                'fenced_code',
                'codehilite',
                'attr_list'
            ])

            # First pass: convert markdown to HTML
            html = md.convert(content)

            # Post-process math blocks
            # Handle display math ($$...$$)
            html = html.replace('<p>$$', '<div class="math-block">$$')
            html = html.replace('$$</p>', '$$</div>')

            # Handle inline math ($...$)
            # We don't need special handling for inline math as MathJax will handle it

            return html
    except Exception as e:
        print(f"Error reading markdown file {filename}: {e}")
        return f"<p>Error loading content: {str(e)}</p>"


@app.route("/get_default_params/<algorithm>")
def get_default_params(algorithm):
    """API endpoint to get default parameters for an algorithm."""
    return get_default_parameters(algorithm)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_default_params/<algorithm>")
def default_params(algorithm):
    # return jsonify(get_default_params(algorithm))
    return create_json_response(get_default_params(algorithm))


@app.route('/md/<filename>')
def get_markdown(filename):
    """Serve markdown files as HTML."""
    if not filename.endswith('.md'):
        filename = filename + '.md'
    html = read_markdown_file(filename)
    return create_json_response({'html': html})


@celery.task(bind=True)
def process_matrices(self, algorithm, params, matrix1_data, matrix2_data):
    try:
        # Convert lists back to numpy arrays
        matrix1 = np.asarray(matrix1_data, dtype=float)
        matrix2 = np.asarray(matrix2_data, dtype=float)
        if matrix1.size == 0 or matrix2.size == 0:
            raise ValueError("Empty matrix received")

        # check if matrices contain NaN, if so, replace with 0
        warning_message = None
        if np.isnan(matrix1).any() or np.isnan(matrix2).any():
            matrix1 = np.nan_to_num(matrix1)
            matrix2 = np.nan_to_num(matrix2)
            warning_message = 'Input matrices contain NaN values. Replaced with 0.'

        # Process based on algorithm
        if algorithm == 'orthogonal':
            result = orthogonal(matrix1, matrix2, **params)
        elif algorithm == 'rotational':
            result = rotational(matrix1, matrix2, **params)
        elif algorithm == 'permutation':
            result = permutation(matrix1, matrix2, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Extract results safely
        if hasattr(result, 't'):
            transformation = result.t
        elif hasattr(result, 't1'):
            transformation = result.t1
        else:
            transformation = np.eye(matrix1.shape[1])

        if hasattr(result, 'new_array'):
            new_array = result.new_array
        elif hasattr(result, 'array_transformed'):
            new_array = result.array_transformed
        else:
            new_array = matrix2

        response_data = {
            'error': float(result.error),
            'transformation': transformation,
            'new_array': new_array
        }

        if warning_message:
            response_data['warning'] = warning_message

        return response_data

    except Exception as e:
        return {'error': f"Processing error: {str(e)}"}


@app.route("/upload", methods=["POST"])
def upload_file():
    print("Received upload request")

    if 'file1' not in request.files or 'file2' not in request.files:
        return create_json_response({'error': 'Both files are required'}, 400)

    file1 = request.files['file1']
    file2 = request.files['file2']
    algorithm = request.form.get('algorithm', 'orthogonal')

    if file1.filename == '' or file2.filename == '':
        return create_json_response({'error': 'No selected files'}, 400)

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return create_json_response({'error': 'Invalid file type'}, 400)

    # Create a unique directory for this upload
    upload_dir = get_unique_upload_dir()

    try:
        # Parse parameters
        try:
            parameters = orjson.loads(request.form.get('parameters', '{}'))
        except orjson.JSONDecodeError:
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

        # Initialize warning message
        warning_message = None

        # Check for NaN values
        if np.isnan(array1).any() or np.isnan(array2).any():
            array1 = np.nan_to_num(array1)
            array2 = np.nan_to_num(array2)
            warning_message = 'Input matrices contain NaN values. Replaced with 0.'

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
            'transformation': transformation,
            'new_array': new_array
        }

        if warning_message:
            response_data['warning'] = warning_message

        return create_json_response(response_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return create_json_response({'error': str(e)}, 500)

    finally:
        # Clean up the unique upload directory
        clean_upload_dir(upload_dir)


@app.route("/status/<task_id>")
def task_status(task_id):
    task = process_matrices.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
        if task.state == 'SUCCESS':
            response['status'] = 'Task completed!'
        else:
            response['status'] = 'Processing...'
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return create_json_response(response)


@app.route("/download", methods=["POST"])
def download():
    try:
        data = orjson.loads(request.form['data'])
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
        return create_json_response({'error': str(e)}, 500)


@app.route('/status')
def server_status():
    """Return server status"""
    status = {
        'status': 'ok',
        'components': {
            'flask': True,
            'celery': False,
            'redis': False
        }
    }

    # Check Celery
    try:
        celery.control.ping(timeout=1)
        status['components']['celery'] = True
    except Exception as e:
        print(f"Celery check failed: {e}")

    # Check Redis
    try:
        redis_client = celery.backend.client
        redis_client.ping()
        status['components']['redis'] = True
    except Exception as e:
        print(f"Redis check failed: {e}")

    # Set overall status based on components
    if not all(status['components'].values()):
        status['status'] = 'degraded'

    return create_json_response(status)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
