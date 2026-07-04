import sys
import os
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# 替换为你的实际模块
from predict_three import load_model, predict_single_image, predict_single_frame, draw_result
from video_thread import VideoThread
from database import *

# 全局配置（美化+优化）
MODEL_PATH = "best_model2.pth"
MAIN_COLOR = "#F9D342"  # 主色调（黄色）
ACCENT_COLOR = "#E67E22"  # 强调色（橙色）
WARNING_1_COLOR = "#E74C3C"  # 一级预警（红色）
WARNING_2_COLOR = "#F39C12"  # 二级预警（橙色）
WARNING_NONE_COLOR = "#27AE60"  # 无预警（绿色）
TEXT_COLOR = "#2C3E50"  # 文字色（深灰蓝）
BG_COLOR = "#FFFFFF"  # 背景色
PANEL_BG_COLOR = "#F8F9FA"  # 面板背景色

# 样式表优化
BUTTON_STYLE = """
    QPushButton {
        background-color: %s;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-size: 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #D35400;
    }
    QPushButton:pressed {
        background-color: #C0392B;
    }
    QPushButton:disabled {
        background-color: #BDC3C7;
        color: #7F8C8D;
    }
""" % ACCENT_COLOR

LABEL_STYLE = """
    QLabel#title_label {
        font-size: 24px;
        font-weight: bold;
        color: %s;
        margin-bottom: 10px;
    }
    QLabel#info_label {
        font-size: 14px;
        color: %s;
        padding: 5px;
    }
    QLabel#result_label {
        font-size: 18px;
        font-weight: bold;
        color: %s;
    }
    QLabel#display_label {
        border: 2px solid %s;
        border-radius: 8px;
        background-color: #F5F5F5;
        font-size: 16px;
        color: #7F8C8D;
    }
    QLabel#warning_label {
        font-size: 14px;
        font-weight: bold;
        padding: 5px;
        border-radius: 4px;
    }
""" % (TEXT_COLOR, TEXT_COLOR, ACCENT_COLOR, MAIN_COLOR)

PANEL_STYLE = f"""
    QWidget#info_panel {{
        background-color: {PANEL_BG_COLOR};
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }}
    QWidget#history_panel {{
        background-color: {PANEL_BG_COLOR};
        border-radius: 8px;
        padding: 15px;
    }}
"""

TABLE_STYLE = f"""
    QTableWidget {{
        border: 1px solid {MAIN_COLOR};
        border-radius: 8px;
        font-size: 14px;
        background-color: white;
        gridline-color: #EEEEEE;
    }}
    QTableWidget::header {{
        background-color: {ACCENT_COLOR};
        color: white;
        font-weight: bold;
        font-size: 14px;
        height: 40px;
    }}
    QTableWidget::item {{
        padding: 8px;
        border: none;
    }}
    QTableWidget::item:selected {{
        background-color: #F1C40F;
        color: {TEXT_COLOR};
    }}
    QScrollBar:vertical {{
        width: 8px;
        background-color: #F5F5F5;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {ACCENT_COLOR};
        border-radius: 4px;
    }}
"""


class CameraThread(QThread):
    """摄像头实时检测线程（已修复信号与绘制）"""
    # 👇 关键修复：信号参数和VideoThread保持一致
    change_pixmap_signal = pyqtSignal(object, str, float)
    status_signal = pyqtSignal(str)

    def __init__(self, model, camera_id=0):
        super().__init__()
        self.model = model
        self.camera_id = camera_id
        self.running = True

    def run(self):
        self.status_signal.emit("正在启动摄像头...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        if not cap.isOpened():
            self.status_signal.emit("摄像头启动失败")
            return

        self.status_signal.emit("正在检测...")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # 👇 修复：调用正确的单帧预测
                pred_class, conf = predict_single_frame(self.model, frame)
                # 👇 修复：必须绘制结果，UI才能显示
                result_frame = draw_result(frame, pred_class, conf)
            except Exception as e:
                print(f"预测错误: {e}")
                result_frame = frame
                pred_class = "error"
                conf = 0.0

            # 👇 关键修复：发射正确参数
            self.change_pixmap_signal.emit(result_frame, pred_class, conf)

        cap.release()
        self.status_signal.emit("检测已停止")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class FogDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于EfficientNet的浓雾等级检测系统")
        self.resize(1400, 900)  # 优化窗口尺寸，避免拥挤
        self.setStyleSheet(f"background-color: {BG_COLOR}; {PANEL_STYLE}")

        # 初始化数据库和模型
        init_db()
        self.model = load_model(MODEL_PATH)

        # 线程管理
        self.video_thread = None
        self.camera_thread = None

        # 状态变量
        self.detection_info = {
            "result": "未检测",
            "confidence": "0.00%",
            "time": "0.00s",
            "warning": "无"
        }

        self.init_ui()

    def init_ui(self):
        """重构UI：解决布局、视觉、交互问题（修正setFixedWidth错误）"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # ---------------------- 左侧：功能区 + 显示区 ----------------------
        left_layout = QVBoxLayout()

        # 标题栏（优化间距）
        title_label = QLabel("基于EfficientNet的浓雾等级检测系统")
        title_label.setObjectName("title_label")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # 功能按钮区（优化分组+状态控制）
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_image = QPushButton("单张图像检测")
        self.btn_video = QPushButton("视频文件检测")
        self.btn_camera = QPushButton("实时摄像头检测")
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setDisabled(True)  # 默认禁用

        # 按钮样式+字体
        btn_font = QFont("Microsoft YaHei", 16)
        for btn in [self.btn_image, self.btn_video, self.btn_camera, self.btn_stop]:
            btn.setFont(btn_font)
            btn.setStyleSheet(BUTTON_STYLE)
            btn_layout.addWidget(btn)

        left_layout.addLayout(btn_layout)
        left_layout.addSpacing(20)  # 增加按钮与展示区间距

        # 检测显示区（优化样式+占位提示）
        self.display_label = QLabel()
        self.display_label.setObjectName("display_label")
        self.display_label.setFixedSize(850, 550)  # 优化尺寸
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet(LABEL_STYLE)
        # 优化占位提示
        self.display_label.setText("""
            <div style="text-align:center;">
                <p>检测画面展示区</p>
                <p style="font-size:14px;color:#BDC3C7;">请选择检测方式开始检测</p>
            </div>
        """)
        left_layout.addWidget(self.display_label, alignment=Qt.AlignCenter)

        main_layout.addLayout(left_layout)

        # ---------------------- 右侧：信息面板 + 历史数据面板（修正setFixedWidth错误） ----------------------
        # 1. 创建右侧容器控件（关键修正：给控件设固定宽度，而非布局）
        right_widget = QWidget()
        right_widget.setFixedWidth(450)  # 给控件设置固定宽度，解决列截断问题
        right_layout = QVBoxLayout(right_widget)  # 布局绑定到控件
        right_layout.setSpacing(20)
        right_layout.setAlignment(Qt.AlignTop)

        # 2. 实时检测信息面板（独立面板+美化）
        info_panel = QWidget()
        info_panel.setObjectName("info_panel")
        info_panel_layout = QVBoxLayout(info_panel)

        info_title = QLabel("实时检测结果")
        info_title.setObjectName("title_label")
        info_panel_layout.addWidget(info_title)

        self.result_label = QLabel(f"检测结果：{self.detection_info['result']}")
        self.result_label.setObjectName("result_label")
        info_panel_layout.addWidget(self.result_label)

        self.conf_label = QLabel(f"置信度：{self.detection_info['confidence']}")
        self.conf_label.setObjectName("info_label")
        info_panel_layout.addWidget(self.conf_label)

        # 预警等级（颜色标注）
        self.warning_label = QLabel(f"预警等级：{self.detection_info['warning']}")
        self.warning_label.setObjectName("warning_label")
        self.update_warning_style("无")  # 初始化样式
        info_panel_layout.addWidget(self.warning_label)

        self.time_label = QLabel(f"用时：{self.detection_info['time']}")
        self.time_label.setObjectName("info_label")
        info_panel_layout.addWidget(self.time_label)

        # 检测状态提示
        self.status_label = QLabel("状态：未开始检测")
        self.status_label.setObjectName("info_label")
        self.status_label.setStyleSheet("color: #7F8C8D;")
        info_panel_layout.addWidget(self.status_label)

        right_layout.addWidget(info_panel)

        # 3. 历史数据面板（独立面板+优化）
        history_panel = QWidget()
        history_panel.setObjectName("history_panel")
        history_panel_layout = QVBoxLayout(history_panel)

        # 历史数据标题+刷新按钮（分组）
        history_header_layout = QHBoxLayout()
        history_title = QLabel("历史检测数据")
        history_title.setObjectName("title_label")
        history_header_layout.addWidget(history_title)

        self.btn_refresh_history = QPushButton("刷新数据")
        self.btn_refresh_history.setStyleSheet(BUTTON_STYLE)
        self.btn_refresh_history.setFont(QFont("Microsoft YaHei", 14))
        self.btn_refresh_history.setFixedSize(100, 35)
        history_header_layout.addWidget(self.btn_refresh_history)
        history_header_layout.addStretch()

        history_panel_layout.addLayout(history_header_layout)

        # 历史数据表格（优化尺寸+排序+列宽）
        self.history_table = QTableWidget()
        self.history_table.setStyleSheet(TABLE_STYLE)
        self.history_table.setColumnCount(5)
        col_labels = ["检测时间", "文件/类型", "雾等级", "置信度", "预警等级"]
        self.history_table.setHorizontalHeaderLabels(col_labels)

        # 优化列宽（避免截断）
        self.history_table.setColumnWidth(0, 120)  # 检测时间
        self.history_table.setColumnWidth(1, 100)  # 文件/类型
        self.history_table.setColumnWidth(2, 70)  # 雾等级
        self.history_table.setColumnWidth(3, 70)  # 置信度
        self.history_table.setColumnWidth(4, 80)  # 预警等级

        # 启用列头排序
        self.history_table.horizontalHeader().setSortIndicatorShown(True)
        self.history_table.horizontalHeader().setSectionsClickable(True)
        self.history_table.horizontalHeader().sectionClicked.connect(self.sort_table)

        # 优化表格尺寸（填充面板）
        self.history_table.setMinimumHeight(300)
        self.history_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        history_panel_layout.addWidget(self.history_table)

        right_layout.addWidget(history_panel)

        # 把右侧容器控件加入主布局（关键修正）
        main_layout.addWidget(right_widget)

        # ---------------------- 绑定按钮事件 ----------------------
        self.btn_image.clicked.connect(self.load_single_image)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_all_detection)
        self.btn_refresh_history.clicked.connect(self.load_history_data)

        # 初始化加载历史数据
        self.load_history_data()

    def update_warning_style(self, warning):
        """优化：预警等级颜色标注"""
        if warning == "一级预警":
            self.warning_label.setStyleSheet(f"""
                QLabel#warning_label {{
                    color: white;
                    background-color: {WARNING_1_COLOR};
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        elif warning == "二级预警":
            self.warning_label.setStyleSheet(f"""
                QLabel#warning_label {{
                    color: white;
                    background-color: {WARNING_2_COLOR};
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        else:
            self.warning_label.setStyleSheet(f"""
                QLabel#warning_label {{
                    color: white;
                    background-color: {WARNING_NONE_COLOR};
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        self.warning_label.setText(f"预警等级：{warning}")

    def update_display(self, frame, is_bgr=True):
        """统一更新显示画面（解决色调问题）"""
        if is_bgr:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pixmap)

    def update_info_panel(self, result, confidence, warning="无", time="0.00s"):
        """更新右侧信息面板（优化提示+预警样式）"""
        self.detection_info["result"] = result
        self.detection_info["confidence"] = f"{confidence:.2f}%"
        self.detection_info["time"] = time

        self.result_label.setText(f"检测结果：{self.detection_info['result']}")
        self.conf_label.setText(f"置信度：{self.detection_info['confidence']}")
        self.update_warning_style(warning)  # 预警样式更新
        self.time_label.setText(f"用时：{self.detection_info['time']}")

    def update_status(self, status):
        """更新检测状态提示"""
        self.status_label.setText(f"状态：{status}")

    def load_single_image(self):
        """单张图像检测"""
        self.stop_all_detection()
        self.update_status("选择图像中...")

        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            self.update_status("未选择图像")
            return

        self.update_status("正在检测...")
        # 检测逻辑
        img, pred, conf = predict_single_image(self.model, path)
        # 计算预警等级
        warning = "无"
        if conf > 60:
            if pred == "1_light_fog":
                warning = "二级预警"
            elif pred == "2_heavy_fog":
                warning = "一级预警"
        insert_record(path, pred, conf, warning)

        # 更新显示和信息
        self.update_display(img, is_bgr=False)
        self.update_info_panel(pred, conf, warning)
        self.update_status("检测完成")
        # 刷新历史数据
        self.load_history_data()

    def load_video(self):
        """视频文件检测（优化交互）"""
        self.stop_all_detection()
        self.update_status("选择视频中...")

        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            self.update_status("未选择视频")
            return

        # 启用停止按钮
        self.btn_stop.setDisabled(False)
        self.update_status("正在加载视频...")

        # 启动视频线程
        self.video_thread = VideoThread(self.model, path)
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        # 临时注释掉这行（解决AttributeError）
        # self.video_thread.status_signal.connect(self.update_status)  # 状态反馈
        self.video_thread.start()

    def update_video_frame(self, frame, pred, conf):
        """统一更新视频/摄像头帧（已自动计算预警）"""
        # 👇 自动计算预警等级（和图片检测逻辑一致）
        warning = "无"
        if conf > 60:
            if pred == "1_light_fog":
                warning = "二级预警"
            elif pred == "2_heavy_fog":
                warning = "一级预警"

        insert_record("camera", pred, conf, warning)
        self.update_display(frame)
        self.update_info_panel(pred, conf, warning)
        # 刷新历史数据（按需刷新，避免性能问题）
        # self.load_history_data()

    def start_camera(self):
        """实时摄像头检测（优化交互）"""
        self.stop_all_detection()
        self.btn_stop.setDisabled(False)
        self.update_status("正在启动摄像头...")

        self.camera_thread = CameraThread(self.model, camera_id=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.start()

    def stop_all_detection(self):
        """停止所有检测线程（优化状态）"""
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        # 停止摄像头线程
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

        # 禁用停止按钮
        self.btn_stop.setDisabled(True)
        # 重置显示和状态
        self.display_label.setText("""
            <div style="text-align:center;">
                <p>检测画面展示区</p>
                <p style="font-size:14px;color:#BDC3C7;">请选择检测方式开始检测</p>
            </div>
        """)
        self.update_info_panel("未检测", 0.0)
        self.update_status("检测已停止")
        # 刷新历史数据
        self.load_history_data()

    def load_history_data(self):
        """加载历史检测数据（优化样式+行高）"""
        self.update_status("加载历史数据...")
        data = query_records()
        self.history_table.setRowCount(len(data))

        for i, row in enumerate(data):
            # row结构：[ID, 时间, 文件名, 雾等级, 置信度, 预警等级]
            self.history_table.setItem(i, 0, QTableWidgetItem(str(row[1])))  # 时间
            self.history_table.setItem(i, 1, QTableWidgetItem(str(row[2])))  # 文件/类型
            self.history_table.setItem(i, 2, QTableWidgetItem(str(row[3])))  # 雾等级
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{float(row[4]):.2f}%"))  # 置信度

            # 预警等级单元格颜色标注
            warning_item = QTableWidgetItem(str(row[5]))
            if row[5] == "一级预警":
                warning_item.setBackground(QColor(WARNING_1_COLOR))
                warning_item.setForeground(QColor("white"))
            elif row[5] == "二级预警":
                warning_item.setBackground(QColor(WARNING_2_COLOR))
                warning_item.setForeground(QColor("white"))
            else:
                warning_item.setBackground(QColor(WARNING_NONE_COLOR))
                warning_item.setForeground(QColor("white"))
            self.history_table.setItem(i, 4, warning_item)

            # 优化行高
            self.history_table.setRowHeight(i, 35)

        # 表头行高
        self.history_table.horizontalHeader().setFixedHeight(40)
        self.update_status("历史数据加载完成")

    def sort_table(self, column):
        """优化：表格列头排序功能"""
        self.history_table.sortByColumn(column,
                                        Qt.AscendingOrder if self.history_table.isSortingEnabled() else Qt.DescendingOrder)

    def closeEvent(self, event):
        """窗口关闭时停止所有线程"""
        self.stop_all_detection()
        event.accept()


# 适配VideoThread（添加状态信号）
"""
修改你的video_thread.py：
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, str, float, tuple)
    status_signal = pyqtSignal(str)  # 新增状态信号

    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.running = True

    def run(self):
        self.status_signal.emit("正在加载视频...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_signal.emit("视频加载失败")
            return

        self.status_signal.emit("正在检测...")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            pred_class, conf = predict_single_frame(self.model, frame)
            warning = "无"
            if conf > 60:
                if pred_class == "1_light_fog":
                    warning = "二级预警"
                elif pred_class == "2_heavy_fog":
                    warning = "一级预警"
            result_frame = draw_result(frame, pred_class, conf)
            self.change_pixmap_signal.emit(result_frame, pred_class, conf, warning, "当前为分类任务，无目标位置信息")
        cap.release()
        self.status_signal.emit("视频检测完成")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 全局字体（优化大小）
    font = QFont("Microsoft YaHei", 14)
    app.setFont(font)

    window = FogDetectionGUI()
    window.show()
    sys.exit(app.exec_())