import ctypes
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import cv2
import numpy as np

"""
stream_handler.py
borrowed from sam3-realtime repo, see:
https://github.com/matteo-tafuro/sam3-realtime/blob/main/scripts/inference/stream_handler.py
"""

class FrameStatus(Enum):
    """Enumeration of frame read outcomes.

    OK: A frame was successfully retrieved.
    NO_FRAME: Source is live/non-blocking and no new frame is currently available (try again later).
    EOS: End-of-stream; no more frames will become available.
    """

    OK = "ok"
    NO_FRAME = "no_frame"
    EOS = "end_of_stream"


@dataclass
class FrameRead:
    """Container for a frame read attempt.

    Attributes:
        status: FrameStatus indicating outcome.
        frame: RGB ndarray (H,W,3) when status == OK, else None.
    """

    status: FrameStatus
    frame: Optional[np.ndarray] = None


class InputStreamHandler:
    """Unified frame source abstraction with explicit read status.

    Supported kinds:
      - video: frames from a video file (blocking); EOS when finished.
      - webcam: frames from a camera device (blocking until failure); EOS if capture ends.
      - yarp: frames from a YARP port (non-blocking); NO_FRAME when no new frame yet.

    read() returns a FrameRead object instead of simply returning raw frame / None.
    This removes ambiguity: None could mean "no frame yet" (YARP) *or* end-of-stream (video/webcam).

    Typical usage:
        src = InputStreamHandler(kind="video", video_path="/path/to/video.mp4")
        src.open()
        while True:
            fr = src.read()
            if fr.status == FrameStatus.NO_FRAME:
                continue  # (Only applies to YARP)
            if fr.status == FrameStatus.EOS:
                break
            frame = fr.frame  # RGB ndarray (H,W,3)
        src.close()
    """

    def __init__(
        self,
        kind: str,
        video_path: Optional[str] = None,
        webcam_index: int = 0,
        fps_request: int = 0,
        skip_n_fr: int = 1,
    ) -> None:
        self.kind = kind
        self.video_path = video_path
        self.webcam_index = webcam_index
        self.fps_request = fps_request

        self._cap: Optional[cv2.VideoCapture] = None
        self.skip_n_fr = max(1, skip_n_fr)
        self.skip_counter = 0

    def open(self) -> None:
        kind = self.kind.lower()
        if kind == "video":
            if not self.video_path or not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video path not found: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self._cap = cap
        elif kind == "webcam":
            cap = cv2.VideoCapture(self.webcam_index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam index {self.webcam_index}")
            self._cap = cap
            if self.fps_request > 0:
                cap.set(cv2.CAP_PROP_FPS, self.fps_request)
        else:
            raise ValueError(f"Unknown source kind: {self.kind}")

    def read(self) -> FrameRead:
        """Attempt to read next RGB frame.
        For video/webcam:
        skip_n_fr=1 -> return every frame
        skip_n_fr=2 -> return every 2nd frame
        skip_n_fr=5 -> return every 5th frame
        Returns:
            FrameRead: (status, frame) where frame is present only if status == FrameStatus.OK.
        """
        kind = self.kind.lower()
        self.skip_counter += 1
        if kind in ("video", "webcam"):
            if self._cap is None:
                return FrameRead(status=FrameStatus.EOS, frame=None)
            ok, bgr = self._cap.read()
            if not ok:
                return FrameRead(status=FrameStatus.EOS, frame=None)
            if self.skip_counter % self.skip_n_fr != 0:
                return FrameRead(status=FrameStatus.NO_FRAME, frame=None)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return FrameRead(status=FrameStatus.OK, frame=rgb)
        else:
            return FrameRead(status=FrameStatus.EOS, frame=None)

    def close(self) -> None:
        kind = self.kind.lower()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Optional convenience: iterator protocol
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def frames(self) -> Iterator[np.ndarray]:
        """Iterator over available frames.

        For YARP sources this will skip NO_FRAME cycles and only yield real frames.
        Terminates on EOS.
        """
        while True:
            fr = self.read()
            if fr.status == FrameStatus.NO_FRAME:
                continue
            if fr.status == FrameStatus.EOS:
                break
            yield fr.frame