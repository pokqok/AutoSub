import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Callable, Optional

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
    def __init__(self, text_threshold: float = 0.3,
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
            cuda=False             # GPU 없으면 False
        )

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def has_text(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        프레임에서 텍스트 영역이 있는지 감지.
        반환: (has_text: bool, bbox: (x1, y1, x2, y2) or None)
        """
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prediction_result = self.craft.detect_text(rgb)
        boxes = prediction_result.get("boxes")

        if boxes is None or len(boxes) == 0:
            return False, None

        # 모든 박스를 감싸는 최소 사각형
        all_pts = np.array(boxes).reshape(-1, 2)
        x1, y1 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        x2, y2 = int(all_pts[:, 0].max()), int(all_pts[:, 1].max())

        # 패딩 추가
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return True, (x1, y1, x2, y2)

    @staticmethod
    def _is_new_subtitle(roi: np.ndarray, prev_roi: Optional[np.ndarray],
                         threshold: float = 0.95) -> bool:
        """이전 ROI와 다르면 True (새 자막)"""
        if prev_roi is None or prev_roi.size == 0:
            return True
        if roi.size == 0:
            return False
        try:
            r1 = cv2.resize(roi, (64, 64))
            r2 = cv2.resize(prev_roi, (64, 64))
            g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
            h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
            h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            return similarity < threshold  # 다르면 새 자막
        except Exception:
            return True  # 비교 실패시 안전하게 새 자막으로 처리

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
                if not self._is_new_subtitle(roi, prev_roi_img, self.dup_threshold):
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
