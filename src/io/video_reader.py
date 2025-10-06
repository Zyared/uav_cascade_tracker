import cv2


class VideoReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")


    def __iter__(self):
        return self


    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise StopIteration
        return frame