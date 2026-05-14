import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Callable, Optional

# ── 모델 캐시를 프로젝트 폴더 내 models/에 저장 (C드라이브 공간 부족 대비) ──
import torch
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_torch_cache = os.path.join(_project_root, "models")
os.makedirs(_torch_cache, exist_ok=True)
torch.hub.set_dir(_torch_cache)

# ── torchvision >= 0.13 호환 패치 ──
# craft-text-detector가 구버전 torchvision의 model_urls를 기대하지만
# torchvision 0.13+ 에서는 삭제됨. 누락된 속성을 주입.
import torchvision.models.vgg as _vgg_mod
if not hasattr(_vgg_mod, 'model_urls'):
    _vgg_mod.model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

from craft_text_detector import Craft


class CRAFTFrameFilter:
    """
    CRAFT 기반 자막 프레임 필터.
    PaddleOCR과 달리 문자 모양 heatmap을 사용하여
    색상/방향/위치에 무관하게 텍스트 영역을 감지합니다.
    """
    def __init__(self, text_threshold: float = 0.2,
                 link_threshold: float = 0.2,
                 low_text: float = 0.4,
                 interval_sec: float = 1.0,
                 dup_threshold: float = 0.95,
                 log_callback: Optional[Callable] = None):
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.interval_sec = interval_sec
        self.dup_threshold = dup_threshold
        self.log_callback = log_callback
        self.craft = None

    def _init_craft(self):
        self.craft = Craft(
            output_dir=None,       # 파일 저장 안 함
            crop_type="poly",
            cuda=True,             # RTX 3050 GPU 사용
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text
        )

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def has_text(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        프레임에서 텍스트 영역이 있는지 감지.
        반환: (has_text: bool, bbox: (x1, y1, x2, y2) or None)
        """
        # GPU 처리용으로 640x360으로 리사이즈
        small = cv2.resize(frame, (640, 360))
        # BGR → RGB
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        try:
            prediction_result = self.craft.detect_text(rgb)
        except ValueError:
            # craft-text-detector 내부에서 inhomogeneous array 버그 발생 시 무시
            return False, None

        if not isinstance(prediction_result, dict):
            return False, None
        boxes = prediction_result.get("boxes")

        if boxes is None or len(boxes) == 0:
            return False, None

        # 모든 박스를 감싸는 최소 사각형 (640x360 기준)
        all_pts = np.array(boxes).reshape(-1, 2)
        x1, y1 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        x2, y2 = int(all_pts[:, 0].max()), int(all_pts[:, 1].max())

        # 640x360 좌표를 원본 frame 해상도로 변환
        h, w = frame.shape[:2]
        fx = w / 640.0
        fy = h / 360.0
        x1 = int(x1 * fx)
        y1 = int(y1 * fy)
        x2 = int(x2 * fx)
        y2 = int(y2 * fy)

        # 패딩 추가 (원본 frame 해상도 기준)
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return True, (x1, y1, x2, y2)

    @staticmethod
    def _is_new_subtitle(roi: np.ndarray, prev_roi: Optional[np.ndarray],
                         threshold: float = 0.03) -> bool:
        """픽셀 diff 기반 — 같은 위치 다른 텍스트도 구분 가능"""
        if prev_roi is None or roi is None or roi.size == 0 or prev_roi.size == 0:
            return True
        try:
            r1 = cv2.resize(roi, (64, 64))
            r2 = cv2.resize(prev_roi, (64, 64))
            diff = cv2.absdiff(r1, r2)
            change_ratio = np.count_nonzero(diff > 10) / diff.size
            return change_ratio > threshold
        except Exception:
            return True

    def filter_frames(self, video_path: str, output_folder: str,
                      progress_callback=None) -> List[Dict]:
        """
        자막이 있는 프레임만 추출하여 output_folder에 저장.
        반환: [{"timestamp": float, "filepath": str, "bbox": (x1,y1,x2,y2)}, ...]
        """
        if self.craft is None:
            self._init_craft()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = max(1, int(fps * self.interval_sec))
        duration = total_frames / fps if fps > 0 else 0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        saved_frames: List[Dict] = []
        prev_roi_img = None
        frame_idx = 0
        saved_count = 0
        skipped_count = 0
        dup_count = 0

        self._log(f"  [CRAFT] Starting: video={os.path.basename(video_path)}, "
                  f"fps={fps:.1f}, interval={self.interval_sec}s, "
                  f"expected_checks≈{int(duration / self.interval_sec)}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % interval_frames != 0:
                    continue

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                # 4K 리사이즈 (CRAFT 처리용)
                h, w = frame.shape[:2]
                scale = 1.0
                max_w = 1920
                if w > max_w:
                    scale = max_w / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                found, bbox = self.has_text(frame)
                if not found:
                    skipped_count += 1
                    continue

                # ROI 크롭
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]

                # 중복 제거
                if not self._is_new_subtitle(roi, prev_roi_img):
                    dup_count += 1
                    continue

                # 저장 (원본 해상도 bbox로 역변환)
                if scale != 1.0:
                    orig_bbox = tuple(int(v / scale) for v in bbox)
                else:
                    orig_bbox = bbox

                timestamp = frame_idx / fps
                filename = f"frame_{timestamp:.3f}.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                saved_frames.append({
                    "timestamp": timestamp,
                    "filepath": filepath,
                    "bbox": orig_bbox,
                    "orig_w": orig_w,
                    "orig_h": orig_h
                })
                saved_count += 1
                prev_roi_img = roi

        finally:
            cap.release()

        self._log(f"  [CRAFT Summary] checked: {frame_idx // interval_frames}, "
                  f"saved: {saved_count}, skipped: {skipped_count}, "
                  f"duplicates removed: {dup_count}, "
                  f"final: {len(saved_frames)}")
        return saved_frames

    def extract_dense_frames(self, video_path: str, markers: List[Dict],
                             output_folder: str,
                             window_sec: float = 2.5,
                             step_sec: float = 0.1,
                             progress_callback=None) -> List[Dict]:
        """
        각 마커 기준 ±window_sec 범위를 step_sec 단위로 추가 샘플링합니다.
        Phase 3(SyncRefiner)에서 이진 탐색할 때 사용됩니다.
        순차 읽기(grab/retrieve)로 seek 비용을 최소화합니다.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"  [Dense] Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 1. 모든 필요 timestamp 수집 + 가장 가까운 마커의 bbox 매핑
        needed: Dict[float, Tuple] = {}  # timestamp -> bbox
        for marker in markers:
            center = float(marker.get("timestamp", 0))
            bbox = marker.get("bbox")
            t = max(0.0, center - window_sec)
            end_t = min(duration, center + window_sec)
            while t <= end_t:
                t_r = round(t, 1)
                if t_r in needed:
                    t += step_sec
                    continue
                needed[t_r] = bbox
                t += step_sec

        if not needed:
            cap.release()
            return []

        # 시간 오름차순으로 정렬된 프레임 목록
        sorted_items = sorted(needed.items(), key=lambda x: x[0])  # [(timestamp, bbox), ...]
        n_needed = len(sorted_items)

        self._log(f"  [Dense] Total needed frames: {n_needed}")

        dense_frames: List[Dict] = []
        count = 0
        current_frame_idx = 0
        next_idx = 0

        while next_idx < n_needed and current_frame_idx < total_frames:
            target_t, target_bbox = sorted_items[next_idx]
            target_frame_idx = round(target_t * fps)

            # grab()으로 필요한 프레임까지 빠르게 건너뛰기 (디코딩 없이)
            while current_frame_idx < target_frame_idx:
                if not cap.grab():
                    break
                current_frame_idx += 1

            if current_frame_idx != target_frame_idx:
                next_idx += 1
                continue

            # target 프레임에 도달했으므로 grab() + retrieve()
            if not cap.grab():
                break
            ret, frame = cap.retrieve()
            current_frame_idx += 1

            if not ret or frame is None:
                next_idx += 1
                continue

            # ROI 크롭 후 저장 (배경 노이즈 제거, SyncRefiner 정밀 비교용)
            if target_bbox:
                x1, y1, x2, y2 = target_bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi_frame = frame[y1:y2, x1:x2]
            else:
                roi_frame = frame

            filepath = os.path.join(output_folder, f"dense_{int(target_t * 1000):08d}.jpg")
            cv2.imwrite(filepath, roi_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            dense_frames.append({
                "timestamp": target_t,
                "filepath": filepath,
                "bbox": (0, 0, roi_frame.shape[1], roi_frame.shape[0]),
                "orig_w": roi_frame.shape[1],
                "orig_h": roi_frame.shape[0]
            })
            count += 1

            if progress_callback:
                progress_callback(count, n_needed)

            next_idx += 1

        cap.release()
        self._log(f"  [Dense] Extracted {count} dense frames ({window_sec}s window, {step_sec}s step)")
        return dense_frames
