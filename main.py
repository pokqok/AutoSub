import sys
import os
import json
import shutil
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar, QComboBox, QListWidget, QAbstractItemView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QSashWindow

from core.sampler import SubtitleSampler
from core.vlm_client import VLMClient
from core.srt_generator import SRTGenerator

CONFIG_FILE = "config.json"
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')

class WorkerThread(QThread):
    progress = Signal(int, int, str) # current, total, status_text
    log = Signal(str)
    finished = Signal(int) # processed count
    error = Signal(str)

    def __init__(self, video_list, settings):
        super().__init__()
        self.video_list = video_list
        self.settings = settings

    def run(self):
        try:
            api_key = self.settings['api_key']
            model_name = self.settings['model_name']
            base_url = self.settings['base_url']
            custom_prompt = self.settings.get('custom_prompt', '')
            
            if not api_key:
                self.error.emit("API Key is missing!")
                return

            client = VLMClient(api_key, model_name, base_url)
            sampler = SubtitleSampler()
            generator = SRTGenerator()
            
            processed_count = 0
            total_videos = len(self.video_list)

            for idx, video_path in enumerate(self.video_list):
                # 1. SRT 존재 여부 확인 (건너뛰기 기능)
                srt_path = os.path.splitext(video_path)[0] + ".srt"
                if os.path.exists(srt_path):
                    self.log.emit(f"Skipping: SRT already exists for {os.path.basename(video_path)}")
                    continue

                self.log.emit(f"Processing: {os.path.basename(video_path)}...")
                
                # 임시 폴더 설정
                temp_dir = os.path.join(os.path.dirname(video_path), "vlm_temp")
                
                # 2. 프레임 추출
                def update_sampling_progress(curr, total):
                    self.progress.emit(curr, total, f"Sampling frames for {os.path.basename(video_path)}...")
                
                sampled_frames = sampler.extract_frames(video_path, temp_dir, progress_callback=update_sampling_progress)
                
                # 3. VLM 분석
                analysis_results = []
                for i, (ts, path) in enumerate(sampled_frames):
                    self.progress.emit(i+1, len(sampled_frames), f"Analyzing frame {i+1}/{len(sampled_frames)}...")
                    res = client.analyze_frame(path, custom_prompt=custom_prompt)
                    if res and res.get('is_dialogue'):
                        analysis_results.append((ts, res.get('translated', '')))
                
                # 4. SRT 생성 및 임시파일 정리
                generator.generate(analysis_results, srt_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                processed_count += 1
                self.log.emit(f"Saved: {srt_path}")

            self.finished.emit(processed_count)

        except Exception as e:
            self.error.emit(str(e))

class SubtitleVLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Subtitle VLM - Japanese to Korean")
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        self.load_settings()
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Settings & Controls
        left_panel = QVBoxLayout()
        
        # Settings Group
        settings_group = QVBoxLayout()
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API Base URL:"))
        self.url_input = QLineEdit(self.settings.get('base_url', 'https://api.deepseek.com'))
        url_layout.addWidget(self.url_input)
        settings_group.addLayout(url_layout)

        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.key_input = QLineEdit(self.settings.get('api_key', ''))
        self.key_input.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.key_input)
        settings_group.addLayout(key_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-v4-pro", "deepseek-v4-flash", "glm-5.1", "gemma-4-31b"])
        self.model_combo.setCurrentText(self.settings.get('model_name', 'deepseek-v4-pro'))
        model_layout.addWidget(self.model_combo)
        settings_group.addLayout(model_layout)
        
        # Custom Prompt Area
        prompt_label = QLabel("Custom Prompt (Jailbreak/Style):")
        settings_group.addWidget(prompt_label)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlainText(self.settings.get('custom_prompt', ''))
        self.prompt_input.setPlaceholderText("Enter instructions to bypass censorship or specify translation style...")
        self.prompt_input.setMaximumHeight(100)
        settings_group.addWidget(self.prompt_input)
        
        left_panel.addLayout(settings_group)
        left_panel.addWidget(QLabel("----------------------------------"))

        # File Queue
        left_panel.addWidget(QLabel("Video Queue (Drag & Drop Folders/Files)"))
        self.video_list_widget = QListWidget()
        left_panel.addWidget(self.video_list_widget)

        btn_add_folder = QPushButton("Add Folder (Recursive)")
        btn_add_folder.clicked.connect(self.add_folder)
        left_panel.addWidget(btn_add_folder)

        btn_clear = QPushButton("Clear List")
        btn_clear.clicked.connect(self.clear_list)
        left_panel.addWidget(btn_clear)

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_process)
        left_panel.addWidget(self.start_btn)

        main_layout.addLayout(left_panel, 1)

        # Right Panel: Logs & Progress
        right_panel = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        right_panel.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        right_panel.addWidget(self.status_label)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        right_panel.addWidget(self.log_window)

        main_layout.addLayout(right_panel, 2)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.add_path(path)

    def add_path(self, path):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(VIDEO_EXTENSIONS):
                        self.video_list_widget.addItem(os.path.join(root, file))
        elif path.lower().endswith(VIDEO_EXTENSIONS):
            self.video_list_widget.addItem(path)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            self.add_path(path)

    def clear_list(self):
        self.video_list_widget.clear()

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                self.settings = json.load(f)
        else:
            self.settings = {}

    def save_settings(self):
        self.settings = {
            "api_key": self.key_input.text(),
            "model_name": self.model_combo.currentText(),
            "base_url": self.url_input.text(),
            "custom_prompt": self.prompt_input.toPlainText()
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f)

    def start_process(self):
        self.save_settings()
        video_files = [self.video_list_widget.item(i).text() for i in range(self.video_list_widget.count())]
        
        if not video_files:
            self.log_window.append("Please add video files to the list first.")
            return

        self.start_btn.setEnabled(False)
        self.log_window.clear()
        
        settings = {
            "api_key": self.key_input.text(),
            "model_name": self.model_combo.currentText(),
            "base_url": self.url_input.text(),
            "custom_prompt": self.prompt_input.toPlainText()
        }

        self.worker = WorkerThread(video_files, settings)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.add_log)
        self.worker.finished.connect(self.process_finished)
        self.worker.error.connect(self.process_error)
        self.worker.start()

    def update_progress(self, curr, total, text):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.status_label.setText(text)

    def add_log(self, message):
        self.log_window.append(message)

    def process_finished(self, count):
        self.start_btn.setEnabled(True)
        self.status_label.setText("Ready")
        self.add_log(f"\n✅ All tasks completed. {count} files processed.")

    def process_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.status_label.setText("Error")
        self.add_log(f"\n❌ Critical Error: {error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleVLMApp()
    window.show()
    sys.exit(app.exec())
