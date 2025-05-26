# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image

# app = Flask(__name__)
# CORS(app)
# # 
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(image_path)

#     try:
#         # Load image
#         image = cv2.imread(image_path)

#         # --- Improved Preprocessing ---

#         # 1) Resize image to increase DPI (try fx=3, fy=3 for better detail)
#         image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

#         # 2) Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # 3) Denoise while preserving edges using bilateral filter
#         denoised = cv2.bilateralFilter(gray, 9, 75, 75)

#         # 4) Adaptive thresholding for binarization (fine-tune blockSize=15, C=3)
#         thresh = cv2.adaptiveThreshold(
#             denoised, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV,
#             15, 3
#         )

#         # 5) Morphological opening (erosion followed by dilation) to reduce noise
#         kernel = np.ones((2, 2), np.uint8)
#         opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

#         # 6) Invert image back to black text on white background
#         processed_image = cv2.bitwise_not(opened)

#         # Convert to PIL Image for pytesseract
#         pil_img = Image.fromarray(processed_image)

#         # --- Tesseract OCR config ---

#         # OEM 1 = LSTM OCR engine (best for handwriting)
#         # PSM 6 = Assume a uniform block of text (you can experiment with 7 or 11)
#         tesseract_config = r'--oem 1 --psm 6'

#         # Perform OCR
#         extracted_text = pytesseract.image_to_string(pil_img, config=tesseract_config)

#         return jsonify({'text': extracted_text.strip()})

#     except pytesseract.TesseractNotFoundError:
#         return jsonify({'error': 'Tesseract OCR not found. Please install it.'}), 500
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#     finally:
#         if os.path.exists(image_path):
#             os.remove(image_path)

# if __name__ == '__main__':
#     app.run(debug=True)


import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps  # Import ImageOps for potential future use, though not strictly needed for simple binarization
import pytesseract

# --- Tesseract OCR Configuration ---
# IMPORTANT: You need to install Tesseract OCR on your system first.
# This is a separate executable program that pytesseract (the Python library) uses.
#
# Installation instructions for Tesseract OCR Engine:
# 1. For Windows:
#    Download the installer from: https://tesseract-ocr.github.io/tessdoc/Home.html
#    Follow the installation wizard. Remember the installation path (e.g., C:\Program Files\Tesseract-OCR).
# 2. For macOS:
#    If you have Homebrew installed, open your Terminal and run:
#    brew install tesseract
# 3. For Linux (Ubuntu/Debian):
#    Open your Terminal and run:
#    sudo apt install tesseract-ocr
#
# If Tesseract is not in your system's PATH, you MUST uncomment and set the path below
# to the directory where the 'tesseract' executable is located.
#
# Since 'tesseract --version' works in your terminal, Tesseract is likely in your system's PATH.
# In this case, pytesseract can find it automatically, and you do NOT need to uncomment
# or set the path below. Keep it commented out as shown:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux/macOS (often not needed if installed via package manager, but useful if you install it manually):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# -----------------------------------

app = Flask(__name__, static_folder='.')  # Serve static files from the current directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            img = Image.open(filepath)
            img = img.convert('L')
            img = img.point(lambda x: 0 if x < 128 else 255)
            tesseract_config = r'--psm 6 --oem 1'
            text = pytesseract.image_to_string(img, lang='eng', config=tesseract_config)
            os.remove(filepath)
            return jsonify({'text': text}), 200

        except pytesseract.TesseractNotFoundError:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Tesseract OCR engine not found. Please ensure it is installed and its path is correctly configured in app.py.'}), 500
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'An error occurred during OCR processing: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
