from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from predict_three import predict_single_frame, draw_result


class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(object, str, float)

    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.running = True

    def run(self):

        cap = cv2.VideoCapture(self.video_path)

        while self.running:

            ret, frame = cap.read()

            if not ret:
                break

            pred_class, conf = predict_single_frame(self.model, frame)

            # 直接在原始BGR图像绘制
            result_frame = draw_result(frame, pred_class, conf)

            self.change_pixmap_signal.emit(result_frame, pred_class, conf)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()