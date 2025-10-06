import cv2


class VideoWriter:
    def __init__(self, path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)


    def write(self, frame):
        self.writer.write(frame)


    def close(self):
        self.writer.release()