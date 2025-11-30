whiteboard-autocropper-blur-detection
Streamlit interface leveraging OpenCV for automated whiteboard detection, contour extraction, and perspective warping. Implements Laplacian variance-based blur detection for image quality assessment. Features adaptive thresholding for enhanced clarity and provides real-time image processing with download functionality. This project is a Streamlit-based web app that automates whiteboard image cropping and enhancement using advanced computer vision techniques. Users upload photos of whiteboards, and the app executes the following steps:

Blur Detection: Uses Laplacian variance (cv2.Laplacian) to assess image sharpness. If blur variance < 100, user is prompted to retake the photo to ensure clarity.

Preprocessing: Converts the image to grayscale and applies Gaussian blur to reduce noise, resizing it for faster processing.

Edge Detection: Applies Canny edge detection (cv2.Canny) to highlight edges.

Contour Detection: Extracts contours using cv2.findContours and identifies the largest 4-point polygon representing the whiteboard.

Perspective Transform: Warps the image to produce a top-down, rectangular view of the whiteboard using cv2.getPerspectiveTransform and cv2.warpPerspective.

Image Enhancement: Applies adaptive thresholding (cv2.adaptiveThreshold) to improve readability of text or drawings on the whiteboard.

UI & Interaction: Streamlit displays the original, contour-highlighted, and final enhanced images side-by-side, with a download button to save the output as PNG. My vision is to extend this web app into a mobile-friendly application for students, enabling effortless capture and digitization of whiteboard notes during lectures, improving note-taking productivity and accessibility.

Installation
To install the required Python packages, run:

pip install -r requirements.txt

Running the App
To launch the Streamlit app, run:

streamlit run app.py

undefined

Live Demo
You can access the live Streamlit app here:
https://white-board-auto-cropper-blur-detection-with-ocr-kb6p4yd67tapf.streamlit.app/
