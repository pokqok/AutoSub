import sys
import os
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal

from core.sampler import SubtitleSampler
from core.vlm_client import VLMClient
from core.srt_generator import SRTGenerator

CONFIG_FILE = "config.json"

class WorkerThread(QThread):
    """
    VLM 자막 생성 파이프라인을 수행하는 백그라운드 스레드
    """
    progress = Signal(int, int) # 현재 프레임, 전체 프레임
    log = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            video_path = self.settings['video_path']
            output_folder = self.settings['output_folder']
            api_key = self.settings['api_key']
            model_name = self.settings['model_name']
            base_url = self.settings['base_url']

            if not video_path or not api_key:
                self.error.emit("비디오 경로와 API 키를 모두 입력해주세요.")
                return

            # 1. 프레임 추출
            self.log.emit("Step 1: 자막 변화 감지 및 프레임 추출 중...")
            sampler = SubtitleSampler()
            frames_dir = os.path.join(output_folder, "temp_frames")
            
            def update_progress(curr, total):
                self.progress.emit(curr, total)

            sampled_frames = sampler.extract_frames(video_path, frames_dir, progress_callback=update_progress)
            self.log.emit(f"총 {len(sampled_frames)}개의 변화 프레임이 검출되었습니다.")

            # 2. VLM 분석 및 번역
            self.log.emit("Step 2: VLM 분석 및 번역 진행 중 (시간이 소요될 수 있습니다)...")
            client = VLMClient(api_key, model_name, base_url)
            analysis_results = []

            for i, (ts, path) in enumerate(sampled_frames):
                res = client.analyze_frame(path)
                if res and res.get('is_dialogue'):
                    translated_text = res.get('translated', '')
                    analysis_results.append((ts, translated_text))
                    self.log.emit(f"[{ts:.2f}s] 추출 성공: {translated_text}")
                else:
                    self.log.emit(f"[{ts:.2f}s] 대사 없음.")
                
                # 진행률 업데이트 (프레임 개수 기준)
                self.progress.emit(i + 1, len(sampled_frames))

            # 3. SRT 파일 생성
            self.log.emit("Step 3: 최종 자막 파일 생성 중...")
            srt_path = os.path.join(output_folder, "output_subtitles.srt")
            generator = SRTGenerator()
            generator.generate(analysis_results, srt_path)

            self.finished.emit(srt_path)

        except Exception as e:
            self.error.emit(f"오류 발생: {str(e)}")

class SubtitleVLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Japanese Subtitle Extractor")
        self.setMinimumSize(700, 500)
        self.load_settings()
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- API 설정 구역 ---
        settings_group = QVBoxLayout()
        
        # API URL
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API Base URL:"))
        self.url_input = QLineEdit(self.settings.get('base_url', 'https://api.deepseek.com'))
        url_layout.addWidget(self.url_input)
        settings_group.addLayout(url_layout)

        # API Key
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.key_input = QLineEdit(self.settings.get('api_key', ''))
        self.key_input.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.key_input)
        settings_group.addLayout(key_layout)

        # Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-v4-pro", "deepseek-v4-flash", "glm-5.1", "gemma-4-31b"])
        self.model_combo.setCurrentText(self.settings.get('model_name', 'deepseek-v4-pro'))
        model_layout.addWidget(self.model_combo)
        settings_group.addLayout(model_layout)

        layout.addLayout(settings_group)
        layout.addWidget(QLabel("-----------------------------------------------------------"))

        # --- 파일 설정 구역 ---
        file_group = QVBoxLayout()

        # 입력 영상
        video_layout = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("영상 파일 경로를 선택하세요...")
        video_layout.addWidget(self.video_input)
        btn_video = QPushButton("찾아보기")
        btn_video.clicked.connect(self.select_video)
        video_layout.addWidget(btn_video)
        file_group.addLayout(video_layout)

        # 출력 폴더
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("결과 저장 폴더를 선택하세요...")
        folder_layout.addWidget(self.folder_input)
        btn_folder = QPushButton("찾아보기")
        btn_folder.clicked.connect(self.select_folder)
        folder_layout.addWidget(btn_folder)
        file_group.addLayout(folder_layout)

        layout.addLayout(file_group)

        # --- 실행 구역 ---
        self.start_btn = QPushButton("자막 추출 시작")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
        self.start_btn.clicked.connect(self.start_process)
        layout.addWidget(self.start_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        layout.addWidget(self.log_window)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "Video Files (*.mp4 *.mkv *.avi)")
        if path:
            self.video_input.setText(path)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if path:
            self.folder_input.setText(path)

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
            "base_url": self.url_input.text()
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f)

    def start_process(self):
        self.save_settings()
        
        settings = {
            "video_path": self.video_input.text(),
            "output_folder": self.folder_input.text(),
            "api_key": self.key_input.text(),
            "model_name": self.model_combo.currentText(),
            "base_url": self.url_input.text()
        }

        self.start_btn.setEnabled(False)
        self.log_window.clear()
        
        self.worker = WorkerThread(settings)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.add_log)
        self.worker.finished.connect(self.process_finished)
        self.worker.error.connect(self.process_error)
        self.worker.start()

    def update_progress(self, curr, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)

    def add_log(self, message):
        self.log_window.append(message)

    def process_finished(self, srt_path):
        self.start_btn.setEnabled(True)
        self.add_log(f"\n✅ 작업 완료! 자막 파일 저장됨: {srt_path}")

    def process_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.add_log(f"\n❌ 오류: {error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleVLMApp()
    window.show()
    sys.exit(app.exec())
