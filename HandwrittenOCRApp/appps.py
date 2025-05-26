import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import pytesseract
import numpy as np # Needed to convert PIL image to OpenCV format
import cv2 # OpenCV library for image processing

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

app = Flask(__name__, static_folder='.') # Serve static files from the current directory
UPLOAD_FOLDER = 'uploads'
# Create the uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to deskew the image using OpenCV
def deskew_image(image_np):
    # Convert image to grayscale for skew detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) # Invert colors for better contour detection

    # Threshold the image to make text black and background white
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))

    # Get the rotated bounding box of the text
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust the angle based on the rectangle's orientation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get the image dimensions
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_np, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Route to serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Route to serve the CSS file
@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

# Route to serve the JavaScript file
@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')

# API endpoint for image upload and OCR processing
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if 'image' file part is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    
    file = request.files['image']
    
    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400
    
    if file:
        # Securely save the uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Open the image using Pillow (PIL)
            img = Image.open(filepath)
            
            # --- Image Pre-processing for OCR ---
            # 1. Rescale to 300 DPI: Tesseract often performs better with higher DPI images.
            current_dpi = img.info.get('dpi', (72, 72)) 
            if current_dpi[0] < 300: 
                scale_factor = 300 / current_dpi[0]
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert PIL Image to OpenCV format (NumPy array) for deskewing
            img_np = np.array(img.convert('RGB')) 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

            # 2. Deskew the image
            img_np = deskew_image(img_np)

            # 3. Convert to grayscale:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            
            # 4. Apply Median Blur for noise reduction:
            #    Median blur is effective for salt-and-pepper noise while preserving edges.
            #    Kernel size must be odd (e.g., 3, 5).
            img_blurred = cv2.medianBlur(img_gray, 3) # Using a 3x3 kernel for subtle smoothing
            
            # 5. Otsu's Binarization:
            #    Automatically finds the optimal threshold value. This is often more robust
            #    than fixed or adaptive thresholding for varied image conditions.
            #    The first return value is the threshold, which we don't need directly.
            _, processed_img_np = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert OpenCV image (NumPy array) back to PIL Image format for pytesseract
            img = Image.fromarray(processed_img_np)
            
# #             # ------------------------------------

            # --- Tesseract Configuration for better accuracy ---
            # --oem 1 : Use LSTM (neural network) OCR engine only, generally better for handwritten text.
            # --psm 3 : Default, assumes a page of text. Good for multi-line text.
            tesseract_config = r'--psm 3 --oem 1' 
            
            # Perform OCR using Tesseract with the pre-processed image and specified configuration
            text = pytesseract.image_to_string(img, lang='eng', config=tesseract_config)

            # Clean up: remove the uploaded file after processing
            os.remove(filepath)
            
            # Return the recognized text as a JSON response
            return jsonify({'text': text}), 200
        
        except pytesseract.TesseractNotFoundError:
            # Handle case where Tesseract executable is not found
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Tesseract OCR engine not found. Please ensure it is installed and its path is correctly configured in app.py.'}), 500
        except Exception as e:
            # Catch any other exceptions during image processing or OCR
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'An error occurred during OCR processing: {str(e)}'}), 500

# Main entry point for the Flask application
if __name__ == '__main__':
    # Run the Flask app. debug=True allows for automatic reloading on code changes
    # and provides more detailed error messages. Set to False for production.
    app.run(debug=True, port=5000) # You can change the port if needed


# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image, ImageOps  # Import ImageOps for potential future use, though not strictly needed for simple binarization
# import pytesseract

# # --- Tesseract OCR Configuration ---
# # IMPORTANT: You need to install Tesseract OCR on your system first.
# # This is a separate executable program that pytesseract (the Python library) uses.
# #
# # Installation instructions for Tesseract OCR Engine:
# # 1. For Windows:
# #    Download the installer from: https://tesseract-ocr.github.io/tessdoc/Home.html
# #    Follow the installation wizard. Remember the installation path (e.g., C:\Program Files\Tesseract-OCR).
# # 2. For macOS:
# #    If you have Homebrew installed, open your Terminal and run:
# #    brew install tesseract
# # 3. For Linux (Ubuntu/Debian):
# #    Open your Terminal and run:
# #    sudo apt install tesseract-ocr
# #
# # If Tesseract is not in your system's PATH, you MUST uncomment and set the path below
# # to the directory where the 'tesseract' executable is located.
# #
# # Since 'tesseract --version' works in your terminal, Tesseract is likely in your system's PATH.
# # In this case, pytesseract can find it automatically, and you do NOT need to uncomment
# # or set the path below. Keep it commented out as shown:
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Example for Linux/macOS (often not needed if installed via package manager, but useful if you install it manually):
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# # -----------------------------------

# app = Flask(__name__, static_folder='.')  # Serve static files from the current directory
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def serve_css():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def serve_js():
#     return send_from_directory('.', 'script.js')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part in the request'}), 400

#     file = request.files['image']

#     if file.filename == '':
#         return jsonify({'error': 'No selected image file'}), 400

#     if file:
#         filename = file.filename
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)

#         try:
#             img = Image.open(filepath)
#             img = img.convert('L')
#             img = img.point(lambda x: 0 if x < 128 else 255)
#             tesseract_config = r'--psm 6 --oem 1'
#             text = pytesseract.image_to_string(img, lang='eng', config=tesseract_config)
#             os.remove(filepath)
#             return jsonify({'text': text}), 200

#         except pytesseract.TesseractNotFoundError:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({'error': 'Tesseract OCR engine not found. Please ensure it is installed and its path is correctly configured in app.py.'}), 500
#         except Exception as e:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({'error': f'An error occurred during OCR processing: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import os
# # import pytesseract
# # import cv2
# # import numpy as np
# # from PIL import Image

# # app = Flask(__name__)
# # CORS(app)

# # # Create a directory for uploads if it doesn't exist
# # UPLOAD_FOLDER = 'uploads'
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # @app.route('/upload', methods=['POST'])
# # def upload_image():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image uploaded'}), 400

# #     file = request.files['image']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     # Save the uploaded image
# #     image_path = os.path.join(UPLOAD_FOLDER, file.filename)
# #     file.save(image_path)

# #     try:
# #         # Load image using OpenCV
# #         image = cv2.imread(image_path)

# #         # Resize image to make text more clear
# #         image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# #         # Convert to grayscale
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #         # Apply Gaussian Blur to reduce noise
# #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# #         # Adaptive thresholding for better text separation
# #         thresh = cv2.adaptiveThreshold(
# #             blurred, 255,
# #             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #             cv2.THRESH_BINARY_INV, 11, 2
# #         )

# #         # Optional: Dilate the image to strengthen characters
# #         kernel = np.ones((2, 2), np.uint8)
# #         dilated = cv2.dilate(thresh, kernel, iterations=1)

# #         # Invert back to black text on white background
# #         processed_image = cv2.bitwise_not(dilated)

# #         # Convert to PIL Image for pytesseract
# #         pil_img = Image.fromarray(processed_image)

# #         # Tesseract config for handwriting: LSTM engine + single line
# #         tesseract_config = r'--oem 1 --psm 7'

# #         # Perform OCR
# #         extracted_text = pytesseract.image_to_string(pil_img, config=tesseract_config)

# #         return jsonify({'text': extracted_text.strip()})

# #     except pytesseract.TesseractNotFoundError:
# #         return jsonify({'error': 'Tesseract OCR not found. Please install it.'}), 500
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500
# #     finally:
# #         # Clean up: remove the saved image
# #         if os.path.exists(image_path):
# #             os.remove(image_path)

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import os
# # import pytesseract
# # import cv2
# # import numpy as np
# # from PIL import Image

# # app = Flask(__name__)
# # CORS(app)

# # UPLOAD_FOLDER = 'uploads'
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # @app.route('/upload', methods=['POST'])
# # def upload_image():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image uploaded'}), 400

# #     file = request.files['image']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     image_path = os.path.join(UPLOAD_FOLDER, file.filename)
# #     file.save(image_path)

# #     try:
# #         # Load image
# #         image = cv2.imread(image_path)

# #         # --- Improved Preprocessing ---

# #         # 1) Resize image to increase DPI (try fx=3, fy=3 for better detail)
# #         image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# #         # 2) Convert to grayscale
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #         # 3) Denoise while preserving edges using bilateral filter
# #         denoised = cv2.bilateralFilter(gray, 9, 75, 75)

# #         # 4) Adaptive thresholding for binarization (fine-tune blockSize=15, C=3)
# #         thresh = cv2.adaptiveThreshold(
# #             denoised, 255,
# #             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #             cv2.THRESH_BINARY_INV,
# #             15, 3
# #         )

# #         # 5) Morphological opening (erosion followed by dilation) to reduce noise
# #         kernel = np.ones((2, 2), np.uint8)
# #         opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# #         # 6) Invert image back to black text on white background
# #         processed_image = cv2.bitwise_not(opened)

# #         # Convert to PIL Image for pytesseract
# #         pil_img = Image.fromarray(processed_image)

# #         # --- Tesseract OCR config ---

# #         # OEM 1 = LSTM OCR engine (best for handwriting)
# #         # PSM 6 = Assume a uniform block of text (you can experiment with 7 or 11)
# #         tesseract_config = r'--oem 1 --psm 6'

# #         # Perform OCR
# #         extracted_text = pytesseract.image_to_string(pil_img, config=tesseract_config)

# #         return jsonify({'text': extracted_text.strip()})

# #     except pytesseract.TesseractNotFoundError:
# #         return jsonify({'error': 'Tesseract OCR not found. Please install it.'}), 500
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500
# #     finally:
# #         if os.path.exists(image_path):
# #             os.remove(image_path)

# # if __name__ == '__main__':
# #     app.run(debug=True)




---------------------------------------------------

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




---------------------------------------------------------------------------


# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# import pytesseract
# import numpy as np
# import cv2

# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')

# def preprocess_image(image_path):
#     """Preprocess image to enhance OCR accuracy."""
#     pil_img = Image.open(image_path).convert('L')
#     img = np.array(pil_img)

#     # Resize to improve OCR on small text
#     img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

#     # Denoise
#     img = cv2.fastNlMeansDenoising(img, h=30)

#     # Adaptive thresholding
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY, 31, 11)

#     # Sharpening
#     sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
#     img = cv2.filter2D(img, -1, sharpen_kernel)

#     return Image.fromarray(img)

# def try_ocr(image, fallback=False):
#     """Perform OCR with optional fallback strategy."""
#     # --psm 6 is good for paragraphs (default), 11 for sparse handwriting
#     config = r'--oem 3 --psm 6' if not fallback else r'--oem 1 --psm 11'
#     return pytesseract.image_to_string(image, lang='eng', config=config)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         # Preprocess the image
#         processed_img = preprocess_image(filepath)

#         # Primary OCR pass
#         text = try_ocr(processed_img).strip()

#         # Fallback if text is too short or failed
#         if len(text) < 10:
#             text = try_ocr(processed_img, fallback=True).strip()

#         os.remove(filepath)
#         return jsonify({'text': text or '[No text detected]'}), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({'error': f'OCR processing error: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



#---------------------------------- Almost Perfect --------------------------------


import os
import io
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torchvision.transforms as transforms
import torch

# === Load TrOCR model once ===
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# === Flask setup ===
app = Flask(__name__, static_folder='.')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Serve frontend static files ===
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')


# === OCR Utility Functions ===

def is_printed_text(image: Image.Image) -> bool:
    """
    Naive check to guess if the handwriting is neat/printed.
    Returns True if printed/typed/neatly written.
    """
    text = pytesseract.image_to_string(image)
    alpha_count = sum(c.isalpha() for c in text)
    noise_ratio = len(text.strip()) / (image.width * image.height)

    # Heuristic: if we get a good amount of clean text and it's not too noisy
    return alpha_count > 30 and noise_ratio < 0.01


def ocr_with_tesseract(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def ocr_with_trocr(image: Image.Image) -> str:
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# === Upload Endpoint ===
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        image = Image.open(filepath)

        # Decide OCR method
        if is_printed_text(image):
            text = ocr_with_tesseract(image)
            source = 'Tesseract'
        else:
            text = ocr_with_trocr(image)
            source = 'TrOCR'

        os.remove(filepath)
        return jsonify({
            'text': text.strip(),
            'engine': source
        }), 200

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500


# === Run Flask Server ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)



# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# import pytesseract
# import numpy as np
# import cv2

# # Configure Flask app
# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Serve static files
# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')

# # Upload and OCR processing route
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected image'}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         # Step 1: Load and convert to grayscale
#         pil_img = Image.open(filepath).convert('L')
#         img = np.array(pil_img)

#         # Step 2: Resize
#         img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

#         # Step 3: Denoise
#         img = cv2.fastNlMeansDenoising(img, h=30)

#         # Step 4: Adaptive Thresholding
#         img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY, 31, 10)

#         # Step 5: Sharpen
#         sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         img = cv2.filter2D(img, -1, sharpen_kernel)

#         # Convert to PIL for Tesseract
#         processed_pil = Image.fromarray(img)

#         # OCR Config tuned for handwriting
#         tesseract_config = r'--oem 1 --psm 11'
#         text = pytesseract.image_to_string(processed_pil, config=tesseract_config)

#         os.remove(filepath)
#         return jsonify({'text': text.strip()}), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({'error': f'OCR processing error: {str(e)}'}), 500

# # Run server
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# import pytesseract
# import numpy as np
# import cv2

# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')

# def preprocess_image(image_path):
#     """Preprocess image to enhance OCR accuracy."""
#     pil_img = Image.open(image_path).convert('L')
#     img = np.array(pil_img)

#     # Resize to improve OCR on small text
#     img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

#     # Denoise
#     img = cv2.fastNlMeansDenoising(img, h=30)

#     # Adaptive thresholding
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY, 31, 11)

#     # Sharpening
#     sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
#     img = cv2.filter2D(img, -1, sharpen_kernel)

#     return Image.fromarray(img)

# def try_ocr(image, fallback=False):
#     """Perform OCR with optional fallback strategy."""
#     # --psm 6 is good for paragraphs (default), 11 for sparse handwriting
#     config = r'--oem 3 --psm 6' if not fallback else r'--oem 1 --psm 11'
#     return pytesseract.image_to_string(image, lang='eng', config=config)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         # Preprocess the image
#         processed_img = preprocess_image(filepath)

#         # Primary OCR pass
#         text = try_ocr(processed_img).strip()

#         # Fallback if text is too short or failed
#         if len(text) < 10:
#             text = try_ocr(processed_img, fallback=True).strip()

#         os.remove(filepath)
#         return jsonify({'text': text or '[No text detected]'}), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({'error': f'OCR processing error: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Load TrOCR model and processor
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # Flask setup
# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Serve static files
# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')

# # Upload and OCR endpoint
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         image = Image.open(filepath).convert("RGB")

#         # Process with TrOCR
#         pixel_values = processor(images=image, return_tensors="pt").pixel_values
#         generated_ids = model.generate(pixel_values)
#         text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         os.remove(filepath)
#         return jsonify({'text': text.strip()}), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({'error': f'TrOCR processing failed: {str(e)}'}), 500

# # Start server
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




# import os
# import io
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# import pytesseract
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import torchvision.transforms as transforms
# import torch

# # === Load TrOCR model once ===
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # === Flask setup ===
# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # === Serve frontend static files ===
# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')


# # === OCR Utility Functions ===

# def is_printed_text(image: Image.Image) -> bool:
#     """
#     Naive check to guess if the handwriting is neat/printed.
#     Returns True if printed/typed/neatly written.
#     """
#     text = pytesseract.image_to_string(image)
#     alpha_count = sum(c.isalpha() for c in text)
#     noise_ratio = len(text.strip()) / (image.width * image.height)

#     # Heuristic: if we get a good amount of clean text and it's not too noisy
#     return alpha_count > 30 and noise_ratio < 0.01


# def ocr_with_tesseract(image: Image.Image) -> str:
#     return pytesseract.image_to_string(image)


# def ocr_with_trocr(image: Image.Image) -> str:
#     transform = transforms.Compose([
#         transforms.Resize((384, 384)),
#         transforms.ToTensor()
#     ])
#     image = image.convert("RGB")
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return text


# # === Upload Endpoint ===
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         image = Image.open(filepath)

#         # Decide OCR method
#         if is_printed_text(image):
#             text = ocr_with_tesseract(image)
#             source = 'Tesseract'
#         else:
#             text = ocr_with_trocr(image)
#             source = 'TrOCR'

#         os.remove(filepath)
#         return jsonify({
#             'text': text.strip(),
#             'engine': source
#         }), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500


# # === Run Flask Server ===
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

# app.py

# import os
# import io
# import cv2
# import numpy as np
# import logging
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # === Setup logging ===
# logging.basicConfig(filename='ocr.log', level=logging.DEBUG, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # === Flask setup ===
# app = Flask(__name__, static_folder='.')
# UPLOAD_FOLDER = 'Uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # === Load TrOCR model ===
# try:
#     processor = TrOCRProcessor.from_pretrained("./trocr_finetuned")
#     model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetuned")
#     logging.info("Loaded fine-tuned TrOCR model.")
# except Exception as e:
#     logging.warning(f"Failed to load fine-tuned model: {e}. Using base model.")
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     logging.info("Loaded base TrOCR model.")

# # === Improved Line Segmentation ===
# def segment_lines(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
#     dilated = cv2.dilate(binary, kernel, iterations=2)

#     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     lines = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if h > 15 and w > 30:
#             y1 = max(y - 10, 0)
#             y2 = min(y + h + 10, image.shape[0])
#             x1 = max(x - 10, 0)
#             x2 = min(x + w + 10, image.shape[1])
#             cropped = image[y1:y2, x1:x2]
#             lines.append((y, cropped))  # Add y to sort top-down

#     # Sort lines top to bottom
#     lines = sorted(lines, key=lambda l: l[0])
#     return [line[1] for line in lines] if lines else [image]

# # === Minimal Preprocessing ===
# def preprocess_image(image):
#     # Convert to RGB if needed
#     if len(image.shape) == 2 or image.shape[2] == 1:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     elif image.shape[2] == 4:
#         image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

#     resized = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
#     return resized

# # === OCR Logic ===
# def ocr_with_trocr(pil_img: Image.Image) -> str:
#     try:
#         image = np.array(pil_img.convert("RGB"))
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         lines = segment_lines(image)
#         logging.info(f"Segmented {len(lines)} line(s).")

#         texts = []
#         for i, line_img in enumerate(lines):
#             processed = preprocess_image(line_img)
#             pixel_values = processor(images=[processed], return_tensors="pt").pixel_values
#             generated_ids = model.generate(pixel_values, max_length=128)
#             text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             logging.debug(f"Line {i+1} OCR: {text}")
#             if text.strip():
#                 texts.append(text)

#         return "\n".join(texts) if texts else "No text recognized."

#     except Exception as e:
#         logging.error(f"OCR failed: {e}")
#         return f"OCR failed: {e}"

# # === Serve frontend ===
# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def style():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def script():
#     return send_from_directory('.', 'script.js')

# # === Upload route ===
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     if not file.content_type.startswith('image/'):
#         return jsonify({'error': 'Please upload an image file'}), 400

#     file.seek(0, os.SEEK_END)
#     if file.tell() > 5 * 1024 * 1024:
#         return jsonify({'error': 'File size exceeds 5MB'}), 400
#     file.seek(0)

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         image = Image.open(filepath)
#         text = ocr_with_trocr(image)
#         os.remove(filepath)
#         logging.info(f"OCR result: {text}")
#         return jsonify({'text': text.strip(), 'engine': 'TrOCR'}), 200

#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         logging.error(f"OCR processing failed: {e}")
#         return jsonify({'error': f'OCR processing failed: {e}'}), 500

# # === Run Flask app ===
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



# import os
# from flask import Flask, request, jsonify, send_from_directory
# from PIL import Image, ImageOps  # Import ImageOps for potential future use, though not strictly needed for simple binarization
# import pytesseract

# # --- Tesseract OCR Configuration ---
# # IMPORTANT: You need to install Tesseract OCR on your system first.
# # This is a separate executable program that pytesseract (the Python library) uses.
# #
# # Installation instructions for Tesseract OCR Engine:
# # 1. For Windows:
# #    Download the installer from: https://tesseract-ocr.github.io/tessdoc/Home.html
# #    Follow the installation wizard. Remember the installation path (e.g., C:\Program Files\Tesseract-OCR).
# # 2. For macOS:
# #    If you have Homebrew installed, open your Terminal and run:
# #    brew install tesseract
# # 3. For Linux (Ubuntu/Debian):
# #    Open your Terminal and run:
# #    sudo apt install tesseract-ocr
# #
# # If Tesseract is not in your system's PATH, you MUST uncomment and set the path below
# # to the directory where the 'tesseract' executable is located.
# #
# # Since 'tesseract --version' works in your terminal, Tesseract is likely in your system's PATH.
# # In this case, pytesseract can find it automatically, and you do NOT need to uncomment
# # or set the path below. Keep it commented out as shown:
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Example for Linux/macOS (often not needed if installed via package manager, but useful if you install it manually):
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# # -----------------------------------

# app = Flask(__name__, static_folder='.')  # Serve static files from the current directory
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def serve_css():
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def serve_js():
#     return send_from_directory('.', 'script.js')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part in the request'}), 400

#     file = request.files['image']

#     if file.filename == '':
#         return jsonify({'error': 'No selected image file'}), 400

#     if file:
#         filename = file.filename
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)

#         try:
#             img = Image.open(filepath)
#             img = img.convert('L')
#             img = img.point(lambda x: 0 if x < 128 else 255)
#             tesseract_config = r'--psm 6 --oem 1'
#             text = pytesseract.image_to_string(img, lang='eng', config=tesseract_config)
#             os.remove(filepath)
#             return jsonify({'text': text}), 200

#         except pytesseract.TesseractNotFoundError:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({'error': 'Tesseract OCR engine not found. Please ensure it is installed and its path is correctly configured in app.py.'}), 500
#         except Exception as e:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({'error': f'An error occurred during OCR processing: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)