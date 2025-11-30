# Whiteboard Cropper, Enhancer & OCR

**A Streamlit app that automatically detects, crops, enhances, and extracts text from whiteboard photos.**  
Built with OpenCV, EasyOCR, NumPy and Streamlit — useful for capturing lecture notes, meeting whiteboards, or printed boards.

---

## Processing pipeline (summary)
1. Upload image → decode with OpenCV.  
2. Preprocess: grayscale → CLAHE → Gaussian blur.  
3. Edge detection (Canny) with multiple thresholds.  
4. Contour detection and filtering (area & aspect ratio).  
5. Perspective transform to obtain a top-down crop.  
6. Artifact removal (Non-local means denoising + morphological opening).  
7. Enhancement (median background subtraction → adaptive threshold).  
8. OCR preprocessing (upscale → bilateral filter → CLAHE → morphology).  
9. OCR with EasyOCR → filter/sort → regex cleanup → display & download.

---

## Demo
1. Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py    
