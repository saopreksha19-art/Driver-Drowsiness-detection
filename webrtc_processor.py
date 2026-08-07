import av
import cv2

from streamlit_webrtc import VideoProcessorBase

from detection import DrowsinessDetector


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = DrowsinessDetector()

        self.status = "Inactive"
        self.score = 0
        self.alarm = False

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        processed, status, score, alarm = self.detector.process_frame(img)

        self.status = status
        self.score = score
        self.alarm = alarm

        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24",
        )