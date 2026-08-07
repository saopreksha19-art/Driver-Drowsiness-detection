import os
import time

import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer


class DrowsinessDetector:

    def __init__(self):

        # Resolve every asset path relative to this file's location,
        # not the process's current working directory. Streamlit can
        # be launched from a different folder than the project root,
        # which otherwise silently breaks these relative paths.
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # -----------------------------
        # Alarm
        # -----------------------------
        try:
            mixer.init()
            self.sound = mixer.Sound(os.path.join(base_dir, "alarm.wav"))
        except Exception:
            self.sound = None
        # -----------------------------
        # Haar Cascades
        # -----------------------------
        cascade_dir = os.path.join(base_dir, "haarcascade")

        self.face = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_alt.xml")
        )

        self.leye = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_lefteye_2splits.xml")
        )

        self.reye = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_righteye_2splits.xml")
        )

        self.eyes = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye.xml")
        )

        # cv2.CascadeClassifier doesn't raise on a bad path - it just
        # loads empty and fails silently on every detectMultiScale
        # call. Fail loudly here instead, at startup.
        for name, classifier in (
            ("face", self.face),
            ("left eye", self.leye),
            ("right eye", self.reye),
            ("eye", self.eyes),
        ):
            if classifier.empty():
                raise RuntimeError(
                    f"Failed to load '{name}' Haar cascade from {cascade_dir}. "
                    "Check that the haarcascade/ folder sits next to detection.py."
                )

        # -----------------------------
        # CNN Model
        # -----------------------------
        self.model = load_model(os.path.join(base_dir, "CNN_m.h5"))
        if self.model is None:
            raise RuntimeError("Failed to load CNN model.")

        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.score = 0
        

        # How much of the frame to scan for a face, as a fraction of
        # full size. Detection runs on this smaller image, then the
        # resulting box is scaled back up. Big speed win, since Haar
        # cascades scan pixel-by-pixel and cost roughly scales with
        # (width * height).
        self.detect_scale = 0.5

        # Wall-clock seconds eyes must be continuously closed before
        # we call it real drowsiness (vs. a normal blink). Using time
        # instead of a frame counter means this threshold means the
        # same thing regardless of your FPS.
        self.closed_eye_threshold_sec = 1.5
        self.eyes_closed_since = None

        # =====================================================
        # Process a Single Camera Frame
        # =====================================================

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # Face (on a downscaled copy — cheaper scan, then coords
        # are rescaled back up to full-frame size)
        # -----------------------------
        s = self.detect_scale
        small_gray = cv2.resize(gray, None, fx=s, fy=s)

        faces = self.face.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25),
        )

        rpred = [1]
        lpred = [1]

        # Track whether we found eyes to test at all this frame, so
        # the closed-eye timer isn't started/reset by a frame where
        # detection simply failed (no face found, bad angle, etc.)
        eyes_tested_this_frame = False

        for (fx, fy, fw, fh) in faces:

            # Scale the downscaled box back up to full-frame coords
            fx, fy, fw, fh = (int(v / s) for v in (fx, fy, fw, fh))

            cv2.rectangle(
                frame,
                (fx, fy),
                (fx + fw, fy + fh),
                (0, 200, 255),
                2,
            )

            # Restrict eye search to inside the face box instead of
            # scanning the whole frame twice more (once per eye
            # cascade). This is the biggest single lag fix here.
            face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

            left_eye = self.leye.detectMultiScale(face_roi_gray)
            right_eye = self.reye.detectMultiScale(face_roi_gray)

            # -----------------------------
            # Right Eye
            # -----------------------------
            if len(right_eye) > 0:
                eyes_tested_this_frame = True

                ex, ey, ew, eh = max(right_eye, key=lambda e: e[2] * e[3])
                # Offset back into full-frame coordinates
                ax, ay = fx + ex, fy + ey

                eye = frame[ay:ay + eh, ax:ax + ew]

                cv2.rectangle(
                    frame,
                    (ax, ay),
                    (ax + ew, ay + eh),
                    (255, 180, 0),
                    2,
                )

                eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                eye_gray = cv2.resize(eye_gray, (48, 48))
                eye_gray = eye_gray.astype(np.float32) / 255.0
                eye_gray = eye_gray.reshape(1, 48, 48, 1)

                prob = self.model.predict(eye_gray, verbose=0)[0][0]
                rpred = [1 if prob >= 0.5 else 0]

            # -----------------------------
            # Left Eye
            # -----------------------------
            if len(left_eye) > 0:
                eyes_tested_this_frame = True

                ex, ey, ew, eh = max(left_eye, key=lambda e: e[2] * e[3])
                ax, ay = fx + ex, fy + ey

                eye = frame[ay:ay + eh, ax:ax + ew]

                cv2.rectangle(
                    frame,
                    (ax, ay),
                    (ax + ew, ay + eh),
                    (255, 180, 0),
                    2,
                )

                eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                eye_gray = cv2.resize(eye_gray, (48, 48))
                eye_gray = eye_gray.astype(np.float32) / 255.0
                eye_gray = eye_gray.reshape(1, 48, 48, 1)

                prob = self.model.predict(eye_gray, verbose=0)[0][0]
                lpred = [1 if prob >= 0.5 else 0]

            # Only need the first detected face
            break

        # -----------------------------
        # Drowsiness Logic (time-based, blink-safe)
        # -----------------------------
        MODEL_OPEN = 1

        right_open = (rpred[0] == MODEL_OPEN)
        left_open = (lpred[0] == MODEL_OPEN)
        eyes_open = right_open and left_open

        if eyes_open or not eyes_tested_this_frame:
            # Either eyes are confirmed open, or we simply couldn't
            # test them this frame (no face found) — don't treat a
            # detection miss as evidence of drowsiness.
            self.eyes_closed_since = None
            status = "Alert"
            self.score = max(self.score - 1, 0)
        else:
            status = "Drowsiness Detected"
            if self.eyes_closed_since is None:
                self.eyes_closed_since = time.time()
            self.score = min(self.score + 1, 30)

        closed_duration = (
            time.time() - self.eyes_closed_since
            if self.eyes_closed_since is not None
            else 0
        )
        alarm = closed_duration >= self.closed_eye_threshold_sec

        if alarm:

            cv2.rectangle(
                frame,
                (0, 0),
                (width, height),
                (0, 60, 255),
                4,
            )

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (35, 55),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                2,
            )

            if self.sound is not None:
                if not mixer.get_busy():
                    self.sound.play()

        else:
            if self.sound is not None:
                self.sound.stop()

        cv2.putText(
            frame,
            status,
            (10, height - 20),
            self.font,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Score: {int(self.score)}",
            (120, height - 20),
            self.font,
            1,
            (255, 255, 255),
            2,
        )

        return frame, status, self.score, alarm

    # =====================================================
    # Release Resources
    # =====================================================

    def close(self):

        if self.sound is not None:
            self.sound.stop()

        try:
            mixer.quit()
        except Exception:
            pass