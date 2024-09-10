from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!doctype html>
<html lang="en">
<head>
  <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
  <meta charset="utf-8">
  <title>Interview Process</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
    }
    #warning {
      color: red;
      font-weight: bold;
      margin-top: 10px;
    }
    #stopFeedButton {
      margin-top: 20px;
      padding: 10px 20px;
      font-size: 16px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h1>Interview Round</h1>
  <video id="videoFeed" width="640" height="480" autoplay></video>
  <div id="warning"></div>
  <button id="stopFeedButton">Stop Feed</button>

  <script>
    const warningDiv = document.getElementById('warning');
    const stopFeedButton = document.getElementById('stopFeedButton');
    const videoFeed = document.getElementById('videoFeed');
    let stream = null;
    let stopFeed = false;

    // Request camera access and stream to video element
    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoFeed.srcObject = stream;
        processVideoStream();  // Start processing the video stream for face detection
      } catch (error) {
        console.error('Error accessing the camera:', error);
        alert('Unable to access camera. Please check your browser settings.');
      }
    }

    // Process video stream frames for face detection
    async function processVideoStream() {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      while (!stopFeed) {
        // Capture the current frame
        canvas.width = videoFeed.videoWidth / 2;  // Resize to half the original width
        canvas.height = videoFeed.videoHeight / 2;  // Resize to half the original height
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(async function(blob) {
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');

          const response = await fetch('/detect_faces', {
            method: 'POST',
            body: formData
          });

          const data = await response.json();
          if (data.num_faces >= 2) {
            warningDiv.textContent = 'Warning: Multiple faces detected!';
          } else {
            warningDiv.textContent = '';
          }
        }, 'image/jpeg', 0.5);  // Compress the image to reduce size

        await new Promise(resolve => setTimeout(resolve, 1000));  // Adjust frame rate to 1 frame per second
      }
    }

    startCamera();

    // Handle stopping the video feed
    stopFeedButton.addEventListener('click', () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());  // Stop the camera stream
        videoFeed.srcObject = null;
        warningDiv.textContent = 'Camera feed stopped';
        stopFeedButton.disabled = true;
        stopFeed = true;
      }
    });
  </script>
</body>
</html>

    """

# Endpoint to detect faces from image frames
@app.post("/detect_faces")
async def detect_faces(image: UploadFile = File(...)):
    try:
        # Read the image file uploaded
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize image for faster processing
        img_resized = cv2.resize(img_np, (320, 240))

        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(40, 40)  # Reduce detection size for faster processing
        )

        num_faces = len(faces)
        return {"num_faces": num_faces}
    except Exception as e:
        logging.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail="Error detecting faces")

# Endpoint to stop the video feed (optional)
@app.post("/terminate")
async def terminate():
    return JSONResponse(content={"success": True})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
