# import json
import inspect
import os
import shutil
import tempfile
import uuid

import markdown
import numpy as np
import orjson
import pandas as pd
# originally use jsonify from flask, but it doesn't support numpy array
from flask import Response, request
from procrustes import orthogonal, permutation, rotational

from celery_config import celery

ALLOWED_EXTENSIONS = {"txt", "npz", "xlsx", "xls"}


__all__ = [
    "allowed_file",
    "get_unique_upload_dir",
    "clean_upload_dir",
    "load_data",
    "save_data",
    "get_default_parameters",
    "create_json_response",
    "read_markdown_file",
]


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

def create_json_response(data, status=200):
    """Create a JSON response using orjson for better numpy array handling"""
    return Response(
        orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY, default=str),
        status=status,
        mimetype="application/json",
    )


def read_markdown_file(filename):
    """Read and convert markdown file to HTML."""
    filepath = os.path.join(os.path.dirname(__file__), "md_files", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Pre-process math blocks to protect them
            # content = content.replace('\\\\', '\\\\\\\\')  # Escape backslashes in math

            # Convert markdown to HTML with math and table support
            md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "attr_list"])

            # First pass: convert markdown to HTML
            html = md.convert(content)

            # Post-process math blocks
            # Handle display math ($$...$$)
            html = html.replace("<p>$$", '<div class="math-block">$$')
            html = html.replace("$$</p>", "$$</div>")

            # Handle inline math ($...$)
            # We don't need special handling for inline math as MathJax will handle it

            return html
    except Exception as e:
        print(f"Error reading markdown file {filename}: {e}")
        return f"<p>Error loading content: {str(e)}</p>"


def get_default_parameters(func: Callable) -> Dict[str, object]:
    """
    Collect the default arguments of a given function as a dictionary.

    Parameters
    ----------
    func : Callable
        The function to inspect.

    Returns
    -------
    Dict[str, object]
        A dictionary where keys are parameter names and values are their default values.

    """
    signature = inspect.signature(func)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
