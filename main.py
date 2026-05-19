import sys
import os
import json
import shutil
import traceback
import time
from typing import List, Tuple, Dict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from core.craft_filter import CRAFTFrameFilter
from core.vlm_client import VLMClient
from core.sync_refiner import SyncRefiner
from core.subtitle_exporter import SubtitleExporter

CONFIG_FILE = "config.json"
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')

class AnalysisWorker(QThread):
    """
    비디오 리스트를 순회하며 Detection 필터 → VLM 배치 → Sync 보정 → 자막 생성을 수행하는 백그라운드 스레드
    """
    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(int, object)
    error = Signal(str)

    def __init__(self, video_list, settings, output_format="SRT"):
        super().__init__()
        self.video_list = video_list
        self.settings = settings
        self.output_format = output_format
        self._cancel = False  # 작업 취소 플래그

    def cancel(self):
        """작업 취소 요청"""
        self._cancel = True
        self.log.emit("Cancellation requested. Stopping after current step...")

    def _check_cancel(self, msg="Operation cancelled by user."):
        """취소 플래그 확인. 취소됐으면 예외 발생"""
        if self._cancel:
            raise InterruptedError(msg)


    def _cleanup_temp_dir(self, temp_dir: str):
        """영상별 임시 폴더 정리"""
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[Cleanup] Removed {temp_dir}")
            except Exception as e:
                print(f"[Cleanup] Failed to remove {temp_dir}: {e}")

    def run(self):
        try:
            api_key = self.settings.get('api_key', '')
            model_name = self.settings.get('model_name', 'gemini-3-flash-preview:cloud')
            base_url = self.settings.get('base_url', '')
            custom_prompt = self.settings.get('custom_prompt', '')

            if not api_key:
                self.error.emit("API Key is missing!")
                return

            client = VLMClient(api_key, model_name, base_url, backup_model=self.settings.get('backup_model_name'))
            self.log.emit("Pre-flight API connection test...")
            self._check_cancel()
            try:
                _ = client.test_connection()
                self.log.emit("  -> API connection verified.")
            except Exception as e:
                self.error.emit(f"API Connection Test Failed:\n{str(e)}\n\n"
                                "Please check your API Key, URL, and Model Name.")
                return

            exporter = SubtitleExporter()
            processed_count = 0
            total_videos = len(self.video_list)
            last_results = []

            for idx, video_path in enumerate(self.video_list):
                self._check_cancel()
                self.log.emit(f"[{idx+1}/{total_videos}] Processing: {os.path.basename(video_path)}")

                ext = ".srt" if self.output_format == "SRT" else ".ass"
                subtitle_path = os.path.splitext(video_path)[0] + ext

                # 임시 폴더
                temp_dir = os.path.join(os.path.dirname(video_path), ".autosub_temp")
                os.makedirs(temp_dir, exist_ok=True)

                try:
                    # Phase 1: CRAFT Detection 필터
                    self._check_cancel()
                    self.log.emit("  Phase 1/3: CRAFT text detection filter...")
                    frame_filter = CRAFTFrameFilter(
                        interval_sec=1.0,
                        log_callback=lambda msg: self.log.emit(msg)
                    )

                    def filter_progress(curr, total):
                        self._check_cancel()
                        pct = int(curr / total * 90)
                        self.progress.emit(pct, 100, f"Phase 1/3: CRAFT filtering {pct}%")

                    subtitle_frames = frame_filter.filter_frames(
                        video_path, temp_dir, progress_callback=filter_progress
                    )
                    self.log.emit(f"  -> {len(subtitle_frames)} subtitle frames detected")
                    self.progress.emit(90, 100, f"Phase 1/3: Extracting dense frames...")

                    if not subtitle_frames:
                        self.log.emit(f"  -> WARNING: No subtitle frames detected in {os.path.basename(video_path)}.")
                        continue

                    # Phase 1 disappear 프레임의 timestamp를 각 마커의 end_ts로 매핑
                    for i, marker in enumerate(subtitle_frames):
                        if marker.get("is_disappear"):
                            continue
                        end_ts = None
                        # 다음 disappear 프레임 찾기
                        for j in range(i + 1, len(subtitle_frames)):
                            if subtitle_frames[j].get("is_disappear"):
                                end_ts = subtitle_frames[j]["timestamp"]
                                break
                        marker["end_ts"] = end_ts

                    # Dense frames for Phase 3 sync refinement
                    self._check_cancel()
                    self.log.emit("  -> Extracting dense frames (start-1.5s ~ end+1.5s @ 0.1s) for sync refinement...")
                    
                    def dense_progress(curr, total):
                        self._check_cancel()
                        pct = 90 + int(curr / total * 10)
                        self.progress.emit(pct, 100, f"Phase 1/3: Dense frames {curr}/{total}")
                    
                    dense_markers = [m for m in subtitle_frames if not m.get("is_disappear")]
                    dense_frames = frame_filter.extract_dense_frames(
                        video_path, dense_markers, temp_dir,
                        window_sec=1.5,
                        progress_callback=dense_progress
                    )
                    self.log.emit(f"  -> Dense frames: {len(dense_frames)}")
                    self.progress.emit(100, 100, f"Phase 1/3: Done ({len(subtitle_frames)} frames, {len(dense_frames)} dense)")
                    all_frames_for_sync = subtitle_frames + dense_frames
                    all_frames_for_sync.sort(key=lambda x: x["timestamp"])

                    # Phase 2: VLM 배치 분석
                    self._check_cancel()
                    self.progress.emit(0, 100, f"Phase 2/3: VLM batch analysis...")
                    self.log.emit(f"  Phase 2/3: VLM batch analysis with '{model_name}'...")
                    BATCH_SIZE = 10
                    all_results = []
                    total_batches = (len(subtitle_frames) + BATCH_SIZE - 1) // BATCH_SIZE

                    for b_idx in range(0, len(subtitle_frames), BATCH_SIZE):
                        self._check_cancel()
                        batch = subtitle_frames[b_idx:b_idx + BATCH_SIZE]
                        batch_num = b_idx // BATCH_SIZE + 1
                        self.log.emit(f"  -> Batch {batch_num}/{total_batches} ({len(batch)} frames)")

                        try:
                            print(f"[DEBUG] frame_batch paths: {[f['filepath'] for f in batch]}")
                            results = client.analyze_batch(
                                batch, custom_prompt=custom_prompt,
                                previous_subtitles=all_results[-3:]
                            )
                            all_results.extend(results)
                            self.log.emit(f"  -> Extracted {len(results)} subtitles from batch {batch_num}")
                        except Exception as e:
                            self.log.emit(f"  -> VLM batch {batch_num} failed: {str(e)}")
                            time.sleep(15)  # Rate limit cooldown before next batch
                            continue

                        pct = int(batch_num / total_batches * 100)
                        self.progress.emit(pct, 100, f"Phase 2/3: Batch {batch_num}/{total_batches} ({pct}%)")

                        # Rate limit avoidance: wait between batches
                        time.sleep(3)

                    self.log.emit(f"  -> VLM total: {len(all_results)} subtitles extracted")
                    if not all_results:
                        self.log.emit(f"  -> WARNING: VLM returned no subtitles for {os.path.basename(video_path)}.")
                        continue

                    print(f"[SORT PRE ] {[(r['start'], r.get('translated','')[:10]) for r in all_results]}")
                    # Phase 2 완료 후, Phase 3 진입 전 — 배치 경계에서 순서 역전 방지
                    all_results.sort(key=lambda x: x["start"])
                    print(f"[SORT POST] {[(r['start'], r.get('translated','')[:10]) for r in all_results]}")

                    # CRAFT end_ts 매핑: VLM의 부정확한 end를 CRAFT disappear timestamp로 보정
                    # 단, 다음 자막 start 이전으로 cap (연속 자막에서 동일 CRAFT end 방지)
                    appear_markers = [m for m in subtitle_frames if not m.get("is_disappear")]
                    for i_cr, result in enumerate(all_results):
                        best_marker = min(appear_markers, key=lambda m: abs(m["timestamp"] - result["start"]))
                        if best_marker.get("end_ts") is not None:
                            craft_end = best_marker["end_ts"]
                            old_end = result["end"]
                            # 다음 자막이 있으면 CRAFT end를 next_start로 cap
                            if i_cr + 1 < len(all_results):
                                next_start = all_results[i_cr + 1]["start"]
                                result["end"] = min(craft_end, next_start - 0.05)
                            else:
                                result["end"] = craft_end
                            print(f"[CRAFT→END] sub start={result['start']:.2f}: VLM end {old_end:.2f} → CRAFT end {craft_end:.2f} → capped {result['end']:.2f}")

                    # Phase 3: 세부 싱크 보정
                    self._check_cancel()
                    self.progress.emit(0, 100, "Phase 3/3: Sync refinement...")
                    self.log.emit("  Phase 3/3: Fine-tuning subtitle sync (0.1s precision)...")
                    try:
                        refiner = SyncRefiner()
                        final_results = refiner.refine(video_path, all_results, all_frames_for_sync)
                        self.log.emit("  -> Sync refinement complete.")
                        self.progress.emit(100, 100, "Phase 3/3: Done")
                    except Exception as e:
                        self.log.emit(f"  -> WARNING: Sync refinement failed, using VLM timing: {str(e)}")
                        final_results = all_results
                        self.progress.emit(100, 100, "Phase 3/3: Done (fallback)")

                    self.progress.emit(95, 100, "Exporting subtitles...")

                    # Phase 4: Post-review (최종 검수)
                    self._check_cancel()
                    self.progress.emit(0, 100, "Phase 4/4: Final review...")
                    self.log.emit("  Phase 4/4: Post-review (QC)...")
                    try:
                        reviewed_results = client.post_review(final_results, custom_prompt=custom_prompt)
                        if len(reviewed_results) != len(final_results):
                            self.log.emit(f"  -> Post-review merged/cleaned subtitles: {len(final_results)} -> {len(reviewed_results)}")
                        final_results = reviewed_results
                        self.log.emit("  -> Post-review complete.")
                        self.progress.emit(100, 100, "Phase 4/4: Done")
                    except Exception as e:
                        self.log.emit(f"  -> WARNING: Post-review failed, keeping Phase 3 result: {str(e)}")
                        self.progress.emit(100, 100, "Phase 4/4: Skipped")


                    # Save
                    self._check_cancel()
                    self.progress.emit(0, 100, "Exporting subtitles...")
                    if self.output_format == "SRT":
                        exporter.generate_srt(final_results, subtitle_path)
                    else:
                        exporter.generate_ass(final_results, subtitle_path)

                    processed_count += 1
                    last_results = final_results
                    self.log.emit(f"Successfully saved {self.output_format}: {os.path.basename(subtitle_path)}")

                finally:
                    # temp 폴더 정리
                    self._cleanup_temp_dir(temp_dir)

            self.finished.emit(processed_count, last_results)

        except InterruptedError as e:
            self.log.emit(f"\n⚠️ Operation cancelled: {e}")
            self.finished.emit(processed_count if 'processed_count' in dir() else 0, [])
        except Exception as e:
            self.error.emit(f"Critical Error: {str(e)}\n{traceback.format_exc()}")

class SubtitleVLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Subtitle VLM Pro - Format Select & Overwrite Confirm")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        self.load_settings()
        self.init_ui()
        self.current_analysis = []
        # 실행때마다 덮어씌워지는 로그 파일
        self.log_file = open("subtitle_vlm.log", "w", encoding="utf-8")
        self.log_file.write(f"=== AutoSub Log - {__import__('datetime').datetime.now().isoformat()} ===\n")
        self.log_file.flush()

    def closeEvent(self, event):
        self.save_settings()
        # 잔여 temp 폴더들 정리
        cleaned = 0
        for i in range(self.video_list_widget.count()):
            vp = self.video_list_widget.item(i).text()
            temp_dir = os.path.join(os.path.dirname(vp), ".autosub_temp")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    cleaned += 1
                except Exception as e:
                    print(f"[Close] Failed to clean temp dir: {e}")
        if cleaned:
            print(f"[Close] Cleaned up {cleaned} temp directories")
        if getattr(self, 'log_file', None):
            self.log_file.close()
        event.accept()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel
        left_panel = QVBoxLayout()
        settings_group = QVBoxLayout()
        
        self.url_input = self.create_setting_row(settings_group, "API Base URL:", self.settings.get('base_url', 'https://api.deepseek.com'))
        self.key_input = self.create_setting_row(settings_group, "API Key:", self.settings.get('api_key', ''), is_password=True)
        
        # Preset 선택 UI
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(150)
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        
        btn_save_preset = QPushButton("💾 Save")
        btn_save_preset.setToolTip("Save current URL/Model/Prompt as a preset (API Key is NEVER saved)")
        btn_save_preset.clicked.connect(self.save_preset_dialog)
        preset_layout.addWidget(btn_save_preset)
        
        btn_del_preset = QPushButton("🗑️ Delete")
        btn_del_preset.clicked.connect(self.delete_preset)
        preset_layout.addWidget(btn_del_preset)
        
        settings_group.addLayout(preset_layout)
        self._refresh_presets()
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Name:"))
        self.model_input = QLineEdit(self.settings.get('model_name', 'gemini-3-flash-preview:cloud'))
        self.model_input.setPlaceholderText("e.g., gemini-3-flash-preview:cloud, gemma3:cloud")
        model_layout.addWidget(self.model_input)
        settings_group.addLayout(model_layout)

        backup_model_layout = QHBoxLayout()
        backup_model_layout.addWidget(QLabel("Backup Model:"))
        self.backup_model_input = QLineEdit(self.settings.get('backup_model_name', ''))
        self.backup_model_input.setPlaceholderText("e.g., gemma4:31b-cloud (fallback on censorship)")
        backup_model_layout.addWidget(self.backup_model_input)
        settings_group.addLayout(backup_model_layout)

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
        
        # 출력 포맷 선택 (Start 버튼 위에 배치)
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Output Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["SRT", "ASS"])
        self.output_format_combo.setCurrentText(self.settings.get('output_format', 'SRT'))
        fmt_layout.addWidget(self.output_format_combo)
        settings_group.addLayout(fmt_layout)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Frame Interval (sec):"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1.0"])
        self.interval_combo.setCurrentText(str(self.settings.get('ocr_interval', 1.0)))
        interval_layout.addWidget(self.interval_combo)
        settings_group.addLayout(interval_layout)
        
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

        btn_test_api = QPushButton("Test API Connection")
        btn_test_api.setStyleSheet("background-color: #1565c0; color: white;")
        btn_test_api.clicked.connect(self.test_api_connection)
        left_panel.addWidget(btn_test_api)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_process)
        btn_layout.addWidget(self.start_btn, 3)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn, 1)
        
        left_panel.addLayout(btn_layout)

        main_layout.addLayout(left_panel, 1)

        # Right Panel
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
        self.edit_table = QTableWidget(0, 5) 
        self.edit_table.setHorizontalHeaderLabels(["Start", "End", "Original", "Translated", "Position"])
        self.edit_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        review_layout.addWidget(self.edit_table)
        
        export_layout = QHBoxLayout()
        self.review_format_combo = QComboBox()
        self.review_format_combo.addItems(["SRT", "ASS"])
        export_layout.addWidget(QLabel("Export Format:"))
        export_layout.addWidget(self.review_format_combo)
        
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

    def _refresh_presets(self):
        """presets.json에서 프리셋 목록을 불러와 콤보박스에 채움"""
        self.preset_combo.clear()
        self.preset_combo.addItem("— Select Preset —")
        presets = self._load_presets()
        for name in sorted(presets.keys()):
            self.preset_combo.addItem(name)
        
    def _load_presets(self) -> dict:
        if os.path.exists("presets.json"):
            try:
                with open("presets.json", "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_presets(self, presets: dict):
        with open("presets.json", "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    
    def save_preset_dialog(self):
        """현재 입력된 URL/모델/프롬프트를 프리셋으로 저장 (API Key는 절대 저장 안 함)"""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._load_presets()
        presets[name] = {
            "base_url": self.url_input.text(),
            "model_name": self.model_input.text(),
            "backup_model_name": self.backup_model_input.text(),
            "custom_prompt": self.prompt_input.toPlainText(),
            "ocr_interval": float(self.interval_combo.currentText()),
            "output_format": self.output_format_combo.currentText()
        }
        self._save_presets(presets)
        self._refresh_presets()
        self.preset_combo.setCurrentText(name)
        self.add_log(f"Preset saved: {name}")
    
    def load_preset(self, name: str):
        """프리셋 선택 시 UI 필드를 채움"""
        if name == "— Select Preset —" or not name:
            return
        presets = self._load_presets()
        p = presets.get(name)
        if not p:
            return
        if p.get("base_url"):
            self.url_input.setText(p["base_url"])
        if p.get("model_name"):
            self.model_input.setText(p["model_name"])
        if p.get("backup_model_name") is not None:
            self.backup_model_input.setText(p["backup_model_name"])
        if p.get("custom_prompt") is not None:
            self.prompt_input.setPlainText(p["custom_prompt"])
        if p.get("ocr_interval"):
            self.interval_combo.setCurrentText(str(p["ocr_interval"]))
        if p.get("output_format"):
            self.output_format_combo.setCurrentText(p["output_format"])
        self.add_log(f"Preset loaded: {name}")
    
    def delete_preset(self):
        """선택된 프리셋 삭제"""
        name = self.preset_combo.currentText()
        if name == "— Select Preset —" or not name:
            return
        presets = self._load_presets()
        if name in presets:
            del presets[name]
            self._save_presets(presets)
            self._refresh_presets()
            self.add_log(f"Preset deleted: {name}")

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
            "model_name": self.model_input.text(),
            "backup_model_name": self.backup_model_input.text(),
            "base_url": self.url_input.text(),
            "custom_prompt": self.prompt_input.toPlainText(),
            "glossary": glossary,
            "output_format": self.output_format_combo.currentText(),
            "ocr_interval": float(self.interval_combo.currentText())
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f)

    def test_api_connection(self):
        """현재 설정값으로 API 연결을 빠르게 테스트합니다."""
        self.save_settings()
        api_key = self.settings.get('api_key', '')
        model_name = self.settings.get('model_name', '')
        base_url = self.settings.get('base_url', '')

        if not api_key:
            QMessageBox.warning(self, "API Test", "API Key is missing!")
            return
        if not base_url:
            QMessageBox.warning(self, "API Test", "API Base URL is missing!")
            return

        self.log_window.append(f"[TEST] Testing API connection to {base_url} with model '{model_name}'...")
        self.status_label.setText("Testing API connection...")
        QApplication.processEvents()

        try:
            client = VLMClient(api_key, model_name, base_url)
            response = client.test_connection()
            self.log_window.append(f"[TEST] ✅ API Connection OK! Response: '{response}'")
            self.status_label.setText("API Connection OK")
            QMessageBox.information(self, "API Test", f"API Connection Successful!\n\nModel: {model_name}\nResponse: '{response}'")
        except Exception as e:
            self.log_window.append(f"[TEST] ❌ API Connection Failed: {str(e)}")
            self.status_label.setText("API Connection Failed")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("API Test Failed")
            msg_box.setIcon(QMessageBox.Critical)
            # 전체 텍스트를 setText에 넣어 한 번에 복사 가능하게
            full_text = f"API Connection Test Failed:\n\n{str(e)}"
            msg_box.setText(full_text)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            msg_box.exec()

    def start_process(self):
        self.save_settings()
        video_files = [self.video_list_widget.item(i).text() for i in range(self.video_list_widget.count())]
        
        if not video_files:
            self.log_window.append("Please add video files to the list first.")
            return

        output_format = self.output_format_combo.currentText()
        ext = ".srt" if output_format == "SRT" else ".ass"
        
        # 덮어쓰기 확인: 이미 자막 파일이 존재하는 영상이 있는지 검사
        existing_files = []
        for vp in video_files:
            sp = os.path.splitext(vp)[0] + ext
            if os.path.exists(sp):
                existing_files.append(os.path.basename(vp))
        
        if existing_files:
            file_list = "\n".join(existing_files[:5])  # 너무 많으면 5개까지만 보여줌
            if len(existing_files) > 5:
                file_list += f"\n... and {len(existing_files) - 5} more"
            
            reply = QMessageBox.question(
                self, 
                "Confirm Overwrite",
                f"The following files already have {output_format} subtitles:\n\n{file_list}\n\nDo you want to overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.log_window.append("Analysis cancelled by user.")
                return

        self.start_btn.setEnabled(False)
        self.log_window.clear()
        
        self.worker = AnalysisWorker(video_files, self.settings, output_format)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.add_log)
        self.worker.finished.connect(self.process_finished)
        self.worker.error.connect(self.process_error)
        self.worker.start()
        self.cancel_btn.setEnabled(True)

    def cancel_process(self):
        """작업 취소: 확인창 → 취소 요청 → 임시 폴더 정리"""
        reply = QMessageBox.question(
            self,
            "Cancel Analysis",
            "Are you sure you want to cancel the current analysis?\n\n"
            "Temporary cache files will be cleaned up.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.add_log("Cancellation requested. Waiting for current step to complete...")
            # 30초까지 대기 후 강제 종료
            self.worker.wait(30000)
            # 잔여 temp 정리 (동일 디렉토리 기준)
            for i in range(self.video_list_widget.count()):
                vp = self.video_list_widget.item(i).text()
                temp_dir = os.path.join(os.path.dirname(vp), ".autosub_temp")
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        self.add_log(f"Cleaned up temp: {os.path.basename(temp_dir)}")
                    except Exception as e:
                        self.add_log(f"Failed to clean temp: {e}")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelled. Ready.")

    def update_progress(self, curr, total, text):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.status_label.setText(text)

    def add_log(self, message):
        self.log_window.append(message)
        if getattr(self, 'log_file', None):
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def process_finished(self, count, results):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(f"Ready. Processed {count} files.")
        self.add_log(f"\n✅ Task completed. {count} files processed.")
        
        self.current_analysis = results
        self.populate_edit_table()
        self.export_btn.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def populate_edit_table(self):
        self.edit_table.setRowCount(0)
        if not self.current_analysis:
            return
        for sub in self.current_analysis:
            row = self.edit_table.rowCount()
            self.edit_table.insertRow(row)
            self.edit_table.setItem(row, 0, QTableWidgetItem(f"{sub['start']:.2f}"))
            self.edit_table.setItem(row, 1, QTableWidgetItem(f"{sub['end']:.2f}"))
            self.edit_table.setItem(row, 2, QTableWidgetItem(sub.get('original', '')))
            self.edit_table.setItem(row, 3, QTableWidgetItem(sub.get('translated', '')))
            self.edit_table.setItem(row, 4, QTableWidgetItem(sub.get('position', 'bottom-center')))

    def export_subtitles(self):
        if not self.video_list_widget.count():
            return
            
        video_path = self.video_list_widget.item(0).text()
        fmt = self.review_format_combo.currentText()
        
        final_subs = []
        glossary = self.settings.get('glossary', {})
        
        for i in range(self.edit_table.rowCount()):
            text = self.edit_table.item(i, 3).text()
            for k, v in glossary.items():
                text = text.replace(k, v)
                
            final_subs.append({
                "start": float(self.edit_table.item(i, 0).text()),
                "end": float(self.edit_table.item(i, 1).text()),
                "translated": text,
                "color": "#FFFFFF"
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
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Error")
        self.add_log(f"\n❌ Critical Error: {error_msg}")
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Analysis Error")
        msg_box.setIcon(QMessageBox.Critical)
        # 전체 텍스트를 setText에 넣어 한 번에 복사 가능하게
        full_text = f"An error occurred during analysis:\n\n{error_msg}"
        msg_box.setText(full_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        # 텍스트를 마우스로 선택/복사 가능하게 설정
        msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        msg_box.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubtitleVLMApp()
    window.show()
    sys.exit(app.exec())
