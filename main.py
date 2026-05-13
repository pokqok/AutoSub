import sys
import os
import json
import shutil
import traceback
from typing import List, Tuple, Dict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal

from core.sampler import SubtitleSampler
from core.vlm_client import VLMClient
from core.subtitle_exporter import SubtitleExporter

CONFIG_FILE = "config.json"
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')

class AnalysisWorker(QThread):
    """
    비디오 리스트를 순회하며 프레임 추출 -> VLM 분석 -> SRT 생성을 수행하는 백그라운드 스레드
    """
    progress = Signal(int, int, str) 
    log = Signal(str)
    # 처리 완료된 파일 개수(int) + 마지막 비디오의 분석 결과(List[Dict])
    finished = Signal(int, object) 
    error = Signal(str)

    def __init__(self, video_list, settings):
        super().__init__()
        self.video_list = video_list
        self.settings = settings

    def run(self):
        try:
            api_key = self.settings.get('api_key', '')
            model_name = self.settings.get('model_name', '')
            base_url = self.settings.get('base_url', '')
            custom_prompt = self.settings.get('custom_prompt', '')

            if not api_key:
                self.error.emit("API Key is missing!")
                return

            client = VLMClient(api_key, model_name, base_url)
            sampler = SubtitleSampler()
            exporter = SubtitleExporter()
            
            processed_count = 0
            total_videos = len(self.video_list)
            last_results = [] # 마지막 비디오의 분석 결과를 저장

            for idx, video_path in enumerate(self.video_list):
                self.log.emit(f"[{idx+1}/{total_videos}] Processing: {os.path.basename(video_path)}")
                
                # SRT 존재 여부 확인 (건너뛰기)
                # 단, 기존 SRT가 비어있거나 실패한 경우(0바이트/대사 없음) 재처리함
                srt_path = os.path.splitext(video_path)[0] + ".srt"
                if os.path.exists(srt_path):
                    try:
                        with open(srt_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        # SRT에 실제 자막 번호와 시간 라인(-->)이 있는지 확인
                        if '-->' in content and len(content) > 50:
                            self.log.emit(f"Skipping: Valid SRT already exists for {os.path.basename(video_path)}")
                            continue
                        else:
                            self.log.emit(f"Re-processing: Existing SRT is empty/invalid for {os.path.basename(video_path)}")
                            os.remove(srt_path)
                    except Exception:
                        # 파일 읽기 실패 시 안전하게 재처리
                        os.remove(srt_path)

                # 임시 프레임 저장 폴더
                temp_dir = os.path.join(os.path.dirname(video_path), "vlm_temp")
                
                # 1. 프레임 추출
                def update_sampling_progress(curr, total):
                    self.progress.emit(curr, total, f"Sampling frames for {os.path.basename(video_path)}...")
                
                sampled_frames = sampler.extract_frames(video_path, temp_dir, progress_callback=update_sampling_progress)
                
                # 2. VLM 분석 및 번역
                self.log.emit(f"Analyzing {len(sampled_frames)} frames...")
                analysis_results = []
                
                glossary = self.settings.get('glossary', {})
                glossary_text = "\n".join([f"{k} -> {v}" for k, v in glossary.items()])
                final_custom_prompt = f"{custom_prompt}\n\nGlossary Guidelines:\n{glossary_text}" if glossary_text else custom_prompt

                for i, (ts, path) in enumerate(sampled_frames):
                    self.progress.emit(i+1, len(sampled_frames), f"Analyzing frame {i+1}/{len(sampled_frames)}...")
                    res = client.analyze_frame(path, custom_prompt=final_custom_prompt)
                    
                    if res and 'dialogues' in res:
                        for dlg in res['dialogues']:
                            analysis_results.append({
                                "start": ts,
                                "end": ts + 1.5,
                                "original": dlg.get('original', ''),
                                "text": dlg.get('translated', ''),
                                "color": dlg.get('color', '#FFFFFF')
                            })
                
                # 3. 타임라인 정제 (중복 제거 및 시간 보정)
                refined_results = []
                if analysis_results:
                    for i in range(len(analysis_results)):
                        curr = analysis_results[i]
                        is_duplicate = False
                        if i > 0:
                            prev = analysis_results[i-1]
                            if prev['text'] == curr['text'] and prev['color'] == curr['color']:
                                refined_results[-1]['end'] = curr['start'] + 1.5
                                is_duplicate = True
                        
                        if not is_duplicate:
                            if i > 0 and refined_results:
                                refined_results[-1]['end'] = curr['start']
                            refined_results.append(curr)

                # 4. SRT 파일 저장
                exporter.generate_srt(refined_results, srt_path)
                
                # 임시 파일 정리
                shutil.rmtree(temp_dir, ignore_errors=True)
                processed_count += 1
                last_results = refined_results # 마지막 결과 갱신
                self.log.emit(f"Successfully saved SRT: {os.path.basename(srt_path)}")

            self.finished.emit(processed_count, last_results)

        except Exception as e:
            self.error.emit(f"Critical Error: {str(e)}\n{traceback.format_exc()}")

class SubtitleVLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Subtitle VLM Pro - Bug Fixed Edition")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        self.load_settings()
        self.init_ui()
        self.current_analysis = []

    def closeEvent(self, event):
        """프로그램 종료 시 설정을 저장합니다."""
        self.save_settings()
        event.accept()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Settings & Queue
        left_panel = QVBoxLayout()
        settings_group = QVBoxLayout()
        
        self.url_input = self.create_setting_row(settings_group, "API Base URL:", self.settings.get('base_url', 'https://api.deepseek.com'))
        self.key_input = self.create_setting_row(settings_group, "API Key:", self.settings.get('api_key', ''), is_password=True)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-v4-pro", "deepseek-v4-flash", "glm-5.1", "gemma-4-31b"])
        self.model_combo.setCurrentText(self.settings.get('model_name', 'deepseek-v4-pro'))
        model_layout.addWidget(self.model_combo)
        settings_group.addLayout(model_layout)

        settings_group.addWidget(QLabel("Custom Prompt (Bypass/Style):"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlainText(self.settings.get('custom_prompt', ''))
        self.prompt_input.setMaximumHeight(100)
        settings_group.addWidget(self.prompt_input)

        settings_group.addWidget(QLabel("Glossary (Key=Value, one per line):"))
        self.glossary_input = QTextEdit()
        glossary_text = "\n".join([f"{k}={v}" for k, v in self.settings.get('glossary', {}).items()])
        self.glossary_input.setPlainText(glossary_text)
        self.glossary_input.setMaximumHeight(100)
        settings_group.addWidget(self.glossary_input)
        
        glossary_btns = QHBoxLayout()
        btn_load_glossary = QPushButton("Load Glossary File")
        btn_load_glossary.clicked.connect(self.load_glossary_file)
        btn_save_glossary = QPushButton("Save Glossary File")
        btn_save_glossary.clicked.connect(self.save_glossary_file)
        glossary_btns.addWidget(btn_load_glossary)
        glossary_btns.addWidget(btn_save_glossary)
        settings_group.addLayout(glossary_btns)
        
        left_panel.addLayout(settings_group)
        left_panel.addWidget(QLabel("----------------------------------"))

        left_panel.addWidget(QLabel("Video Queue"))
        self.video_list_widget = QListWidget()
        left_panel.addWidget(self.video_list_widget)
        
        file_btns = QHBoxLayout()
        btn_add_file = QPushButton("Add File")
        btn_add_file.clicked.connect(self.add_single_file)
        btn_add_folder = QPushButton("Add Folder")
        btn_add_folder.clicked.connect(self.add_folder)
        file_btns.addWidget(btn_add_file)
        file_btns.addWidget(btn_add_folder)
        left_panel.addLayout(file_btns)

        btn_remove_file = QPushButton("Remove Selected")
        btn_remove_file.clicked.connect(self.remove_selected_file)
        left_panel.addWidget(btn_remove_file)

        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_process)
        left_panel.addWidget(self.start_btn)

        main_layout.addLayout(left_panel, 1)

        # Right Panel: Logs & Review
        right_panel = QVBoxLayout()
        self.tabs = QTabWidget()

        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        log_layout.addWidget(self.progress_bar)
        log_layout.addWidget(self.status_label)
        log_layout.addWidget(self.log_window)
        self.tabs.addTab(self.log_tab, "Process Log")

        self.review_tab = QWidget()
        review_layout = QVBoxLayout(self.review_tab)
        self.edit_table = QTableWidget(0, 4) 
        self.edit_table.setHorizontalHeaderLabels(["Start", "End", "Text", "Color(HEX)"])
        self.edit_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        review_layout.addWidget(self.edit_table)
        
        export_layout = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["SRT", "ASS"])
        export_layout.addWidget(QLabel("Export Format:"))
        export_layout.addWidget(self.format_combo)
        
        self.export_btn = QPushButton("Export Subtitles")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_subtitles)
        export_layout.addWidget(self.export_btn)
        review_layout.addLayout(export_layout)
        self.tabs.addTab(self.review_tab, "Edit & Review")

        right_panel.addWidget(self.tabs)
        main_layout.addLayout(right_panel, 2)

    def create_setting_row(self, parent_layout, label_text, default_val, is_password=False):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        line_edit = QLineEdit(default_val)
        if is_password: line_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(line_edit)
        parent_layout.addLayout(layout)
        return line_edit

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.add_path(url.toLocalFile())

    def add_path(self, path):
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(VIDEO_EXTENSIONS):
                        self.video_list_widget.addItem(os.path.join(root, file))
        elif path.lower().endswith(VIDEO_EXTENSIONS):
            self.video_list_widget.addItem(path)

    def add_single_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.flv *.wmv)")
        if path:
            self.video_list_widget.addItem(path)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: self.add_path(path)

    def remove_selected_file(self):
        """선택된 항목을 큐에서 제거합니다."""
        selected_items = self.video_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.video_list_widget.takeItem(self.video_list_widget.row(item))

    def load_glossary_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Glossary File", "", "Text Files (*.txt);;JSON Files (*.json)")
        if not path: return
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        text = "\n".join([f"{k}={v}" for k, v in data.items()])
                        self.glossary_input.setPlainText(text)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    self.glossary_input.setPlainText(f.read())
        except Exception as e:
            self.log_window.append(f"Error loading glossary: {e}")

    def save_glossary_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Glossary File", "", "Text Files (*.txt);;JSON Files (*.json)")
        if not path: return
        try:
            text = self.glossary_input.toPlainText()
            if path.endswith('.json'):
                glossary = {}
                for line in text.split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        glossary[k.strip()] = v.strip()
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(glossary, f, ensure_ascii=False, indent=2)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
        except Exception as e:
            self.log_window.append(f"Error saving glossary: {e}")

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f: self.settings = json.load(f)
        else: self.settings = {}

    def save_settings(self):
        glossary = {}
        for line in self.glossary_input.toPlainText().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                glossary[k.strip()] = v.strip()
        
        self.settings = {
            "api_key": self.key_input.text(),
            "model_name": self.model_combo.currentText(),
            "base_url": self.url_input.text(),
            "custom_prompt": self.prompt_input.toPlainText(),
            "glossary": glossary
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
        
        self.worker = AnalysisWorker(video_files, self.settings)
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

    def process_finished(self, count, results):
        """분석 완료 후 결과를 Review 탭에 채웁니다."""
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"Ready. Processed {count} files.")
        self.add_log(f"\n✅ Task completed. {count} files processed.")
        
        # Review 탭에 결과 연결
        self.current_analysis = results
        self.populate_edit_table()
        self.export_btn.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def populate_edit_table(self):
        """현재 분석 결과를 Edit & Review 테이블에 표시합니다."""
        self.edit_table.setRowCount(0)
        if not self.current_analysis:
            return
        for sub in self.current_analysis:
            row = self.edit_table.rowCount()
            self.edit_table.insertRow(row)
            self.edit_table.setItem(row, 0, QTableWidgetItem(f"{sub['start']:.2f}"))
            self.edit_table.setItem(row, 1, QTableWidgetItem(f"{sub['end']:.2f}"))
            self.edit_table.setItem(row, 2, QTableWidgetItem(sub['text']))
            self.edit_table.setItem(row, 3, QTableWidgetItem(sub.get('color', '#FFFFFF')))

    def export_subtitles(self):
        """테이블에서 편집된 데이터를 읽어 자막 파일을보냅니다."""
        if not self.video_list_widget.count():
            return
            
        video_path = self.video_list_widget.item(0).text()
        fmt = self.format_combo.currentText()
        
        final_subs = []
        glossary = self.settings.get('glossary', {})
        
        for i in range(self.edit_table.rowCount()):
            text = self.edit_table.item(i, 2).text()
            # 후처리 단어집 기반 치환
            for k, v in glossary.items():
                text = text.replace(k, v)
                
            final_subs.append({
                "start": float(self.edit_table.item(i, 0).text()),
                "end": float(self.edit_table.item(i, 1).text()),
                "text": text,
                "color": self.edit_table.item(i, 3).text()
            })
        
        exporter = SubtitleExporter()
        if fmt == "SRT":
            out_path = os.path.splitext(video_path)[0] + ".srt"
            exporter.generate_srt(final_subs, out_path)
        else:
            out_path = os.path.splitext(video_path)[0] + ".ass"
            exporter.generate_ass(final_subs, out_path)
            
        self.add_log(f"Exported to {out_path}")

    def process_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.status_label.setText("Error")
        self.add_log(f"\n❌ Critical Error: {error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleVLMApp()
    window.show()
    sys.exit(app.exec())
