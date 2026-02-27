import os
import cv2
import base64
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from roboflow import Roboflow
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

# ==============================
# 1. CONFIGURATION & INITIALIZATION
# ==============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

API_KEY = "jGSGNfJd9TV1Nyj0RctA"  
LANDSLIDE_PROJECT = "segformer-landslide-detection"
LANDSLIDE_VERSION = 2
LANDMINE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print("⏳ Loading Roboflow model...")
try:
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(LANDSLIDE_PROJECT)
    landslide_model = project.version(LANDSLIDE_VERSION).model
    print("✅ Roboflow model loaded")
except Exception as e:
    print(f"⚠️ Failed to load Roboflow model: {e}")

print("⏳ Loading YOLO model...")
landmine_model = None
if os.path.exists(LANDMINE_MODEL_PATH):
    landmine_model = YOLO(LANDMINE_MODEL_PATH)
    print("✅ Landmine YOLO model loaded")
else:
    print(f"⚠️ Landmine model not found at {LANDMINE_MODEL_PATH}")

box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.0)

# ==============================
# 2. PROCESSING LOGIC
# ==============================
def process_image(image_path, filename):
    raw_frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_frame is None:
        return None, "Failed to load image."

    # Normalize image to strictly 3-channel BGR for consistent processing
    if len(raw_frame.shape) == 2:
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR) 
    elif len(raw_frame.shape) == 3 and raw_frame.shape[2] == 4:
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR) 
    else:
        frame = raw_frame.copy()

    h, w = frame.shape[:2]
    annotated = frame.copy()
    
    # --------------------------------
    # PIXEL-PERFECT GRAYSCALE CHECK
    # --------------------------------
    # Split channels to mathematically check for color presence
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    
    # Calculate average pixel difference between channels
    diff_bg = np.mean(np.abs(b.astype(int) - g.astype(int)))
    diff_gr = np.mean(np.abs(g.astype(int) - r.astype(int)))
    
    # If the channels are nearly identical (difference < 3 allows for tiny JPEG artifacts), it's grayscale
    is_rgb = not (diff_bg < 3.0 and diff_gr < 3.0)

    landslide_found = False
    landmine_found = False
    mode_used = ""

    # --------------------------------
    # AUTO-ROUTING LOGIC
    # --------------------------------
    if is_rgb:
        mode_used = "Mode: Color Image -> Landslide Scan"
        print(f"🔍 {mode_used} ({filename})")
        try:
            result = landslide_model.predict(image_path).json()
            predictions = result.get("predictions", [])
            overlay = np.zeros_like(frame)
            mask_combined = np.zeros((h, w), dtype=np.uint8)

            for pred in predictions:
                mask_str = pred.get("segmentation_mask")
                if mask_str:
                    landslide_found = True
                    mask_bytes = base64.b64decode(mask_str)
                    mask_array = np.frombuffer(mask_bytes, np.uint8)
                    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (w, h))
                    mask_combined[mask > 0] = 255

            if landslide_found:
                overlay[mask_combined > 0] = (0, 165, 255)
                annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.5, 0)
                contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, (0, 100, 255), 3)
        except Exception as e:
            print("⚠️ Landslide API error:", e)

    else:
        mode_used = "Mode: Grayscale -> Landmine Scan"
        print(f"🔍 {mode_used} ({filename})")
        if landmine_model:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm_result = landmine_model(rgb_frame, conf=0.25, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(lm_result)

            if len(detections) > 0:
                landmine_found = True
                labels = [
                    f"{landmine_model.names[class_id]} {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # --------------------------------
    # TWO-LINE PRESENTATION CANVAS
    # --------------------------------
    status = []
    if landslide_found: status.append("LANDSLIDE")
    if landmine_found: status.append("LANDMINE")

    status_text = " | ".join(status) + " DETECTED" if status else "SAFE - NO THREATS DETECTED"
    bg_color = (0, 0, 200) if status else (0, 200, 0) 
    text_color = (255, 255, 255)

    padding = max(20, int(w * 0.04))
    header_height = max(100, int(h * 0.15)) 
    
    final_w = w + (padding * 2)
    final_h = h + header_height + (padding * 2)
    max_text_width = final_w - (padding * 2) # Strict boundary to stop text cut-off

    presentation_canvas = np.full((final_h, final_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(presentation_canvas, (0, 0), (final_w, header_height), bg_color, -1)

    # 1. Scale Top Line (Status)
    font_scale_status = max(0.6, final_w / 1200)
    thickness_status = max(1, int(font_scale_status * 2.5))
    size_status = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale_status, thickness_status)[0]
    
    if size_status[0] > max_text_width:
        font_scale_status = font_scale_status * (max_text_width / size_status[0])
        size_status = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale_status, thickness_status)[0]

    # 2. Scale Bottom Line (Mode Info)
    font_scale_mode = max(0.4, final_w / 1800)
    thickness_mode = max(1, int(font_scale_mode * 2))
    size_mode = cv2.getTextSize(mode_used, cv2.FONT_HERSHEY_DUPLEX, font_scale_mode, thickness_mode)[0]

    if size_mode[0] > max_text_width:
        font_scale_mode = font_scale_mode * (max_text_width / size_mode[0])
        size_mode = cv2.getTextSize(mode_used, cv2.FONT_HERSHEY_DUPLEX, font_scale_mode, thickness_mode)[0]

    # Calculate vertical spacing
    gap = int(header_height * 0.15) 
    total_text_height = size_status[1] + gap + size_mode[1]
    start_y = (header_height - total_text_height) // 2
    
    # Draw Status Text
    text_x_status = (final_w - size_status[0]) // 2
    text_y_status = start_y + size_status[1]
    cv2.putText(presentation_canvas, status_text, (text_x_status, text_y_status),
                cv2.FONT_HERSHEY_DUPLEX, font_scale_status, text_color, thickness_status, cv2.LINE_AA)

    # Draw Mode Text
    text_x_mode = (final_w - size_mode[0]) // 2
    text_y_mode = text_y_status + gap + size_mode[1]
    cv2.putText(presentation_canvas, mode_used, (text_x_mode, text_y_mode),
                cv2.FONT_HERSHEY_DUPLEX, font_scale_mode, text_color, thickness_mode, cv2.LINE_AA)

    # Paste the image
    y_offset = header_height + padding
    x_offset = padding
    presentation_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = annotated
    cv2.rectangle(presentation_canvas, (x_offset-2, y_offset-2), (x_offset+w+1, y_offset+h+1), (80, 80, 80), 2)

    result_filename = f"processed_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, presentation_canvas)

    # Return only the main status to the frontend (frontend hides the mode logic inside its banner automatically)
    return result_filename, status_text

# ==============================
# 3. FLASK ROUTES
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file.")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_filename, status_text = process_image(filepath, filename)
            
            if not result_filename:
                return render_template('index.html', error=status_text)

            result_url = url_for('static', filename=f'results/{result_filename}')
            original_url = url_for('static', filename=f'uploads/{filename}')
            
            return render_template('index.html', 
                                   result_url=result_url, 
                                   original_url=original_url,
                                   status_text=status_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)