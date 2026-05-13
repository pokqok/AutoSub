import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple, Callable

class PaddleVLExtractor:
    """
    PaddleOCR-VL-1.5 (0.9B) 로컬 GPU 추출기.
    transformers 라이브러리를 통해 로드하고, 영상 프레임에서 텍스트를 추출합니다.
    """
    def __init__(self, model_path: str = "PaddlePaddle/PaddleOCR-VL-1.5",
                 device: Optional[str] = None, interval_sec: float = 0.3,
                 similarity_threshold: float = 0.6,
                 log_callback: Optional[Callable] = None):
        self.model_path = model_path
        self.interval_sec = interval_sec
        self.similarity_threshold = similarity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_callback = log_callback
        self._model = None
        self._processor = None

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

    def _extract_text_from_frame(self, pil_image: Image.Image, frame_idx: int = 0) -> str:
        """단일 PIL 이미지에서 텍스트를 추출합니다."""
        if self._model is None:
            self._load_model()

        # 더 단순하고 명확한 프롬프트 (문서용 태그 제거)
        prompt = "Extract all Japanese text visible in this image. Return only the text lines, separated by newlines. Do not add any explanation."
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
        
        # 프롬프트 및 잡음 제거
        clean_text = generated_text.strip()
        if clean_text.lower().startswith("extract all"):
            # 프롬프트가 echo된 경우
            lines = clean_text.split("\n")
            # 첫 줄이 프롬프트 echo면 제거
            if lines and ("extract" in lines[0].lower() or "image" in lines[0].lower()):
                lines = lines[1:]
            clean_text = "\n".join(lines).strip()
        
        if frame_idx <= 5:
            self._log(f"  [VL Debug frame {frame_idx}] raw output: '{generated_text[:100]}...' -> clean: '{clean_text[:80]}...'")
        
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

                # 하단 60%만 crop해서 보냄 (자막은 보통 하단, 문서 모델은 전체 화면에 산만함)
                h, w = frame.shape[:2]
                crop_top = int(h * 0.4)
                cropped = frame[crop_top:h, 0:w]
                
                rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                try:
                    current_raw = self._extract_text_from_frame(pil_image, frame_idx)
                except Exception as e:
                    self._log(f"  [VL Error frame {frame_idx}] {str(e)}")
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
        return results
