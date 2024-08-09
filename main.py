from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import cv2
import numpy as np
from typing import AsyncIterator, Tuple
import threading
import uvicorn
import logging
import asyncio 

app = FastAPI()


# Initialize Haar Cascade classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
glasses_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Flag and lock to control the video feed
stop_video_feed = threading.Event()
video_feed_lock = threading.Lock()

# Set up logging
logging.basicConfig(level=logging.INFO)

def detect_objects(frame: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        global count
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 10, minSize=(40, 40))
        num_faces = len(faces)

        if num_faces >= 2:
            cv2.putText(frame, 'Warning: Multiple faces detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        

        return frame, num_faces
    except Exception as e:
        logging.error(f"Error in detect_objects: {e}")
        return frame, 0

def detect_objects(frame: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 10, minSize=(40, 40))
        num_faces = len(faces)
        return frame, num_faces
    except Exception as e:
        logging.error(f"Error in detect_objects: {e}")
        return frame, 0

async def generate_frames() -> AsyncIterator[bytes]:
    while not stop_video_feed.is_set():
        try:
            with video_feed_lock:
                success, frame = video_capture.read()
                if not success:
                    logging.warning("Failed to capture frame from video source.")
                    break
                frame, num_faces = detect_objects(frame)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                if num_faces >= 2:
                    frame_bytes += b'\n--alert--'  # Append alert marker to the frame bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            break
        await asyncio.sleep(0.1)  # Yield control back to the event loop


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!doctype html>
<html lang="en">
<head>
  <link rel="icon" href="/favicon.ico" type="image/x-icon">
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
  <img src="/video_feed" width="640" height="480" id="videoFeed" alt="Live Video Feed">
  <div id="warning"></div>
  <button id="stopFeedButton">Stop Test</button>
  
  <script>
    const warningDiv = document.getElementById('warning');
    const stopFeedButton = document.getElementById('stopFeedButton');
    const videoFeed = document.getElementById('videoFeed');

    stopFeedButton.addEventListener('click', async () => {
      try {
        const response = await fetch('/terminate', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
          alert('Test stopped successfully.');
          // Hide the video feed and stop updating the image
          videoFeed.src = '';
          warningDiv.textContent = '';
          stopFeedButton.disabled = true;  // Disable the button to prevent further clicks
        } else {
          alert('Failed to stop the feed.');
        }
      } catch (error) {
        console.error('Error stopping the feed:', error);
        alert('An error occurred while stopping the feed.');
      }
    });
  </script>
</body>
</html>
    """

@app.get("/video_feed")
async def video_feed():
    try:
        return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Error in video_feed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/terminate")
async def terminate():
    try:
        # Set the event to stop the video feed
        stop_video_feed.set()
        
        # Ensure the video capture is properly released
        video_capture.release()
        #cv2.destroyAllWindows()
        
        # Return a success response
        return JSONResponse(content={"success": True})
    except Exception as e:
        logging.error(f"Error during termination: {e}")
        # Return an error response
        raise HTTPException(status_code=500, detail="Failed to terminate video feed")

if __name__ == "_main_":
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    try:
        server_thread.join()
    except KeyboardInterrupt:
        stop_video_feed.set()
        video_capture.release()
        cv2.destroyAllWindows()
