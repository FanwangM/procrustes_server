from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
import procrustes
import tempfile
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'npz', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(filepath):
    """Load data from various file formats."""
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'npz':
        with np.load(filepath) as data:
            return data['arr_0'] if 'arr_0' in data else next(iter(data.values()))
    elif ext == 'txt':
        return np.loadtxt(filepath)
    elif ext in ['xlsx', 'xls']:
        df = pd.read_excel(filepath)
        return df.to_numpy()

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
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received upload request")  # Debug log
    if 'file1' not in request.files or 'file2' not in request.files:
        print("Missing files in request")  # Debug log
        return jsonify({'error': 'Both files are required'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    algorithm = request.form.get('algorithm', 'orthogonal')

    # Parse additional parameters
    try:
        parameters = {}
        if request.form.get('parameters'):
            parameters = json.loads(request.form.get('parameters'))
            print(f"Additional parameters: {parameters}")  # Debug log
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON in parameters field'}), 400

    print(f"Algorithm selected: {algorithm}")  # Debug log
    print(f"File names: {file1.filename}, {file2.filename}")  # Debug log

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected files'}), 400

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save files temporarily
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
        file1.save(file1_path)
        file2.save(file2_path)

        print(f"Files saved: {file1_path}, {file2_path}")  # Debug log

        # Load data
        array1 = load_data(file1_path)
        array2 = load_data(file2_path)

        print(f"Arrays loaded - shapes: {array1.shape}, {array2.shape}")  # Debug log

        # Perform Procrustes analysis with additional parameters
        if algorithm == 'orthogonal':
            result = procrustes.orthogonal(array1, array2, **parameters)
        elif algorithm == 'rotational':
            result = procrustes.rotational(array1, array2, **parameters)
        elif algorithm == 'permutation':
            result = procrustes.permutation(array1, array2, **parameters)
        else:
            return jsonify({'error': 'Invalid algorithm'}), 400

        print(f"Analysis completed - error: {result.error}")  # Debug log
        print(f"Result attributes: {dir(result)}")  # Debug log

        # Clean up temporary files
        os.remove(file1_path)
        os.remove(file2_path)

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
            print(f"Error in response preparation: {str(e)}")
            return jsonify({'error': 'Error preparing response data'}), 500

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug log
        import traceback
        print(traceback.format_exc())  # Print full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_result():
    try:
        data = json.loads(request.form['data'])
        format_type = request.form['format']

        if format_type not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid format type'}), 400

        array_data = np.array(data)
        result_file = save_data(array_data, format_type)

        return send_file(
            result_file,
            as_attachment=True,
            download_name=f'procrustes_result.{format_type}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
