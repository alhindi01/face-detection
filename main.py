import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
from io import BytesIO
import streamlit as st
import numpy as np
from PIL import Image, ImageColor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

detection_mode = None
# Haar-Cascade Parameters
minimum_neighbors = 4
# Minimum possible object size
min_object_size = (50, 50)
# bounding box thickness
# bounding box color
bbox_color = (0, 255, 0)

def main():
    

    # -------------Header Section------------------------------------------------   

    title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Smart Monitoring Using Artificial Intelligence</p>'
    st.markdown(title, unsafe_allow_html=True)  

    supported_modes = "<html> " \
                      "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
                      "<ul><li>Image Upload Then Face Detection</li><li>Webcam Image Capture Then Face Detection</li><li>Face Detection Real-time</li><li>Object Detection Real-time</li></ul>" \
                      "</div></body></html>"
    st.markdown(supported_modes, unsafe_allow_html=True)    

    st.info("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")

    object_detection_page = "Real Time Object detection"

    image_upload_page = ("Upload An Image"
    )

    webcame_image_capture_page = "Capture An Image from Webcam"


    webcam_realtime_page = "Real Time Face Detection"

    programatically_control_page = "Control the playing state programatically"

    with st.sidebar:
        st.image("./assets/faceman_cropped.png", width=260)

        title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
        st.markdown(title, unsafe_allow_html=True)

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            
            image_upload_page,
            webcame_image_capture_page,
            webcam_realtime_page,
            object_detection_page


        ],
        
    )
    
    st.subheader(app_mode)



    if app_mode == image_upload_page:
        image_upload()
    elif app_mode == object_detection_page:
        app_object_detection()
    elif app_mode == webcame_image_capture_page:
        webcame_image_capture()
    elif app_mode == webcam_realtime_page:
        webcam_realtime()



    # -------------Image Upload Section------------------------------------------------


def image_upload():
    st.markdown(
        "Upload an image, then Frontal-face detection using OpenCV and Haar-Cascade Algorithm will be applied to it"
    )    

    # Example Images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/example_2.png")
    with col2:
        st.image(image="./assets/example_3.png")

    uploaded_file = st.file_uploader(
        "Upload Image (Only PNG & JPG images allowed)", type=['png', 'jpg'])

    if uploaded_file is not None:

        with st.spinner("Detecting faces..."):
            img = Image.open(uploaded_file)

            # To convert PIL Image to numpy array:
            img = np.array(img)

            # Load the cascade
            face_cascade = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')

            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

            if len(faces) == 0:
                st.warning(
                    "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting."
                    " Also try adjusting detection parameters")
            else:
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  color=bbox_color)

                # Display the output
                st.image(img)

                if len(faces) > 1:
                    st.success("Total of " + str(
                        len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")

                    # convert to pillow image
                    img = Image.fromarray(img)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")

                    # Creating columns to center button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col3:
                        pass
                    with col2:
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")
                else:
                    st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")

                    # convert to pillow image
                    img = Image.fromarray(img)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")

                    # Creating columns to center button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col3:
                        pass
                    with col2:
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")

# -------------Webcam Image Capture Section------------------------------------------------
def webcame_image_capture():
    st.markdown(
        "Capture a picture from your webcam, or your screen and then Frontal-face detection using OpenCV and Haar-Cascade Algorithm will be applied to it "
    ) 
    
    st.info("NOTE : In order to use this mode, you need to give webcam access.")

    img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                      help="Make sure you have given webcam permission to the site")

    if img_file_buffer is not None:

        with st.spinner("Detecting faces ..."):
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img = np.array(img)

            # Load the cascade
            face_cascade = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')

            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=minimum_neighbors,
                                                  minSize=min_object_size)

            if len(faces) == 0:
                st.warning(
                    "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting. "
                    "Also try adjusting detection parameters")
            else:
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  color=bbox_color)

                # Display the output
                st.image(img)

                if len(faces) > 1:
                    st.success("Total of " + str(
                        len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")
                else:
                    st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything")

                # Download the image
                img = Image.fromarray(img)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                # Creating columns to center button
                col1, col2, col3 = st.columns(3)
                with col1:
                    pass
                with col3:
                    pass
                with col2:
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")

# -------------Webcam Realtime Section------------------------------------------------


def webcam_realtime():
    st.markdown(
        "Enable your webcam, then Real-Time Frontal-face detection using OpenCV and Haar-Cascade Algorithm will be applied to it"
    )    
    # load face detection model
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    st.info("NOTE : In order to use this mode, you need to give webcam access. "
               )

    spinner_message = "Wait a sec, getting some things done..."

    with st.spinner(spinner_message):

        class VideoProcessor:

            def recv(self, frame):
                # convert to numpy array
                frame = frame.to_ndarray(format="bgr24")

                # detect faces
                faces = cascade.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 3)

                # draw bounding boxes
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)

                frame = av.VideoFrame.from_ndarray(frame, format="bgr24")

                return frame

        webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                        rtc_configuration=RTCConfiguration(
                            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
    

# -------------object detection Section------------------------------------------------

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_object_detection():

    st.markdown(
        "Object detection with MobileNet SSD Algorithm"
)    

    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    # -------------Hide Streamlit Watermark------------------------------------------------

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

    main()