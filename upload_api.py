import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# --- CONFIGURATION FOR UPLOAD API ---
UPLOAD_FOLDER = 'incoming_videos_for_analysis' # This is where uploaded videos will go
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'} # Add common video formats
MAX_FILE_SIZE_MB = 500 # Set a generous file size limit, e.g., 500 MB

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024 # Convert MB to bytes

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging Setup ---
log_format = '[%(asctime)s.%(msecs).03d] [%(name)s,%(funcName)s:%(lineno)s] [%(levelname)s] %(message)s'
logging.basicConfig(filename='upload_api.log', level=logging.INFO, format=log_format, datefmt='%d/%b/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.path}, Remote Addr: {request.remote_addr}")

@app.route('/upload-video', methods=['POST'])
def upload_video():
    logger.info("Received request for video upload.")
    # Check if a file part is in the request
    if 'file' not in request.files:
        logger.error("No 'file' part in the request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if a file was actually selected
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    # Process allowed file
    if file and allowed_file(file.filename):
        # Secure the filename and add a timestamp to make it unique
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        unique_filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            logger.info(f"File '{original_filename}' saved successfully as '{unique_filename}' to {filepath}")
            return jsonify({
                "status": "success",
                "message": "Video uploaded successfully!",
                "uploaded_filename": unique_filename,
                "filepath": filepath
            }), 200
        except Exception as e:
            logger.error(f"Error saving file '{original_filename}': {e}", exc_info=True)
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    else:
        logger.error(f"File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/ping', methods=['GET'])
def ping():
    timestamp = datetime.utcnow().isoformat()
    logger.info(f"Ping request received. Timestamp: {timestamp}")
    return jsonify({"timestamp": timestamp}), 200

if __name__ == '__main__':
    logger.info("Starting simple video upload API.")
    app.run(host='0.0.0.0', port=5003, debug=True)