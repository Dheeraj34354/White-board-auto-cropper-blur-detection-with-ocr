import re
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import easyocr

# Cache EasyOCR reader
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (5, 5), 0)

def find_whiteboard_contour(edges, shape):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = shape[:2]
    min_area = 0.05 * w * h
    max_area = 0.99 * w * h
    best, max_area_found = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area and area > max_area_found:
            rect = cv2.minAreaRect(c)
            W, H = rect[1]
            if W > 0 and H > 0:
                aspect = max(W, H) / (min(W, H) + 1e-8)
                if 0.3 < aspect < 4.0:
                    best, max_area_found = c, area
    return best

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = order_points(box)
    W, H = int(rect[1][0]), int(rect[1][1])
    if W < 20 or H < 20:
        return None
    dst_pts = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M = cv2.getPerspectiveTransform(box.astype('float32'), dst_pts)
    return cv2.warpPerspective(image, M, (W, H))

def remove_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    return opened

def enhance_image(gray):
    background = cv2.medianBlur(gray, 51)
    diff = cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 13, 7)
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

def resize_for_display(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(image, (max_width, int(h * ratio)), interpolation=cv2.INTER_LANCZOS4)
    return image

def clean_ocr_text(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9\-\.,:\(\)\s]', '', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

st.title("Whiteboard Cropper, Enhancer & OCR")
uploaded_file = st.file_uploader("Upload a whiteboard photo", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if original_image is None:
        st.error("Cannot decode uploaded image.")
    else:
        h, w = original_image.shape[:2]
        st.header("1. Original Image")
        st.image(cv2.cvtColor(resize_for_display(original_image), cv2.COLOR_BGR2RGB), channels="RGB")

        pre = preprocess_image(original_image)
        cropped = None
        for t1, t2 in [(10, 50), (50, 150), (100, 200)]:
            edges = cv2.Canny(pre, t1, t2)
            contour = find_whiteboard_contour(edges, original_image.shape)
            if contour is not None:
                tmp = perspective_transform(original_image, contour)
                if tmp is not None and tmp.shape[0] > 0 and tmp.shape[1] > 0:
                    cropped = tmp
                    break

        if cropped is None:
            st.info("No board contour detected — using safe central crop.")
            pad = int(min(h, w) * 0.05)
            cropped = original_image[pad:h-pad, pad:w-pad]

        st.header("2. Cropped Whiteboard Region")
        st.image(cv2.cvtColor(resize_for_display(cropped), cv2.COLOR_BGR2RGB), channels="RGB")

        cleaned_gray = remove_artifacts(cropped)
        result = enhance_image(cleaned_gray)

        st.header("3. Final Enhanced Image")
        st.image(cv2.cvtColor(resize_for_display(result), cv2.COLOR_BGR2RGB), channels="RGB")

        # Improved OCR preprocessing + EasyOCR call
        st.header("Extracted Text from Image")
        with st.spinner("Extracting text (improved)..."):
            # upscale and denoise
            scale = 3
            gray_up = cv2.resize(cleaned_gray, (cleaned_gray.shape[1]*scale, cleaned_gray.shape[0]*scale),
                                 interpolation=cv2.INTER_CUBIC)
            gray_up = cv2.bilateralFilter(gray_up, d=9, sigmaColor=75, sigmaSpace=75)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_up = clahe.apply(gray_up)

            # adaptive threshold (handwriting-friendly)
            th = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
            if np.mean(th) < 127:
                th = 255 - th

            # morphological close then slight erosion to join strokes and thin thick lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
            th = cv2.erode(th, kernel, iterations=1)

            ocr_img = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

            try:
                raw = reader.readtext(ocr_img, detail=1, paragraph=False,
                                      contrast_ths=0.05, adjust_contrast=0.7,
                                      text_threshold=0.35, low_text=0.3, link_threshold=0.4,
                                      mag_ratio=1)
            except Exception:
                raw = []

            # filter low-confidence, sort top-to-bottom
            filtered = [r for r in raw if r[2] >= 0.30]
            filtered.sort(key=lambda x: min(pt[1] for pt in x[0]))

            # group words into lines by vertical proximity
            lines = []
            prev_y = None
            y_tol = 30 * scale
            for bbox, text, conf in filtered:
                top_y = min(pt[1] for pt in bbox)
                if prev_y is None or abs(top_y - prev_y) > y_tol:
                    lines.append(text)
                else:
                    lines[-1] += " " + text
                prev_y = top_y

            cleaned_lines = [clean_ocr_text(l) for l in lines]
            extracted_text = "\n".join(cleaned_lines) if cleaned_lines else "[No text detected]"

        st.text_area("Copy the text below", value=extracted_text, height=240)

        # Download enhanced image
        img_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Enhanced PNG", buf, "enhanced_whiteboard_final.png", mime="image/png")
else:
    st.markdown("Upload a whiteboard image to get started!")