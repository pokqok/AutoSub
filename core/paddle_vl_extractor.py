import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple, Callable
import os

class PaddleVLExtractor:
    """
    PaddleOCR-VL-1.5 (0.9B) 로컬 GPU 추출기.
    transformers 라이브러리를 통해 로드하고, 영상 프레임에서 텍스트를 추출합니다.
    """
    def __init__(self, model_path: str = "PaddlePaddle/PaddleOCR-VL-1.5",
                 device: Optional[str] = None, interval_sec: float = 0.3,
                 similarity_threshold: float = 0.6,
                 log_callback: Optional[Callable] = None,
                 debug_dir: Optional[str] = None):
        self.model_path = model_path
        self.interval_sec = interval_sec
        self.similarity_threshold = similarity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_callback = log_callback
        self.debug_dir = debug_dir  # 디버깅용 프레임 저장 폴더
        self._model = None
        self._processor = None
        self._debug_frame_count = 0

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def _save_debug_image(self, pil_image: Image.Image, prefix: str, text: str):
        """디버깅용 이미지 저장"""
        if not self.debug_dir:
            return
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_'))
            filename = f"{prefix}_{self._debug_frame_count:04d}_{safe_text}.png"
            filepath = os.path.join(self.debug_dir, filename)
            pil_image.save(filepath)
            self._debug_frame_count += 1
        except Exception:
            pass

    def _extract_text_from_frame(self, pil_image: Image.Image, frame_idx: int = 0) -> str:
        """단일 PIL 이미지에서 텍스트를 추출합니다."""
        if self._model is None:
            self._load_model()
            self._log(f"  [VL] Model loaded on {self.device}")

        # 애니메이션/영상 자막에 특화된 프롬프트 (문서용 태그 제거, 매우 구체적)
        prompt = (
            "Look at this image from a Japanese anime or video. "
            "Find any Japanese text (hiragana, katakana, kanji) shown as subtitles at the bottom or anywhere on screen. "
            "Return ONLY the Japanese text. No explanations, no translations, no markdown. "
            "If there is no Japanese text, say exactly: NO_TEXT"
        )
        
        inputs = self._processor(
            text=prompt,
            images=pil_image.convert("RGB"),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # 프롬프트 echo 및 잡음 제거
        clean_text = generated_text.strip()
        
        # 첫 줄이 프롬프트 관련 내용이면 제거
        lines = clean_text.split("\n")
        filtered_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if any(skip in line_lower for skip in [
                "look at this", "find any japanese", "return only", 
                "no explanations", "if there is no", "this image from"
            ]):
                continue
            if line_lower == "no_text":
                return ""
            filtered_lines.append(line)
        
        clean_text = "\n".join(filtered_lines).strip()
        
        # 디버깅 이미지 저장
        self._save_debug_image(pil_image, f"frame_{frame_idx}", clean_text)
        
        if frame_idx <= 10:
            self._log(f"  [VL Debug frame {frame_idx}] raw: '{generated_text[:120]}' clean: '{clean_text[:80]}'")
        
        return clean_text

    def extract(self, video_path: str, progress_callback=None) -> List[Dict]:
        """영상에서 자막을 추출하여 [{start, end, original, position}] 반환"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = max(1, int(fps * self.interval_sec))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from difflib import SequenceMatcher

        results: List[Dict] = []
        buffered_text = ""
        start_time = -1.0
        frame_idx = 0
        last_timestamp = 0.0
        total_text_frames = 0
        debug_dir = os.path.join(os.path.dirname(video_path), "vl_debug_frames")
        self.debug_dir = debug_dir

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % interval_frames == 0:
                curr_t = frame_idx / fps
                last_timestamp = curr_t
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                # 하단 60% crop (자막은 보통 하단)
                h, w = frame.shape[:2]
                crop_top = int(h * 0.4)
                cropped = frame[crop_top:h, 0:w]
                
                rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                try:
                    current_raw = self._extract_text_from_frame(pil_image, frame_idx)
                except Exception as e:
                    self._log(f"  [VL Error frame {frame_idx}] {str(e)}")
                    import traceback
                    self._log(traceback.format_exc())
                    current_raw = ""

                if current_raw.strip():
                    total_text_frames += 1

                # 버퍼링 / 병합 로직
                if current_raw != buffered_text:
                    if buffered_text != "" and (
                        current_raw == "" or
                        SequenceMatcher(None, current_raw, buffered_text).ratio() < self.similarity_threshold
                    ):
                        if buffered_text.strip():
                            results.append({
                                "start": start_time,
                                "end": curr_t,
                                "original": buffered_text.strip(),
                                "position": "bottom-center"
                            })
                        start_time = curr_t if current_raw != "" else -1.0

                    buffered_text = current_raw

        cap.release()

        # 마지막 버퍼 Flush
        if buffered_text and buffered_text.strip():
            if start_time < 0:
                start_time = last_timestamp
            results.append({
                "start": start_time,
                "end": last_timestamp,
                "original": buffered_text.strip(),
                "position": "bottom-center"
            })

        self._log(f"  [VL Summary] total frames checked: {frame_idx // interval_frames}, "
                  f"frames with text: {total_text_frames}, final segments: {len(results)}")
        self._log(f"  [VL Debug] Debug frames saved to: {debug_dir}")
        return results
