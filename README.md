# AutoSub - AI 하드코딩 자막 추출·번역기

CRAFT 검출 + 멀티모달 VLM + 연속 프레임 diff 스파이크 탐색 파이프라인으로, 영상 속 **하드코딩 일본어 자막**을 검출해 **한국어 자막(SRT / ASS)**으로 변환하는 Windows GUI 애플리케이션입니다.

## 🎯 핵심 개념

**"CRAFT가 자막 프레임을 뽑고 → VLM이 이미지로 읽어 한꺼번에 번역 → 연속 프레임 diff로 정확한 등장/사라짐 감지"**

### 파이프라인

| 단계 | 파일 | 역할 |
|------|------|------|
| **Phase 1** | `core/craft_filter.py` | 1초 간격 프레임 샘플링 → CRAFT로 자막 영역 검출 → 픽셀 diff 기반 중복 제거 → `timestamp`, `bbox`, `filepath` 메타데이터 + 등장/소멸 주변 dense frame(±1.0초) 추출 |
| **Phase 2** | `core/vlm_client.py` | 최대 10장 배치로 멀티모달 VLM 전달. 일본어 원문 → 한국어 번역, 색상, 위치, 타이밍 추정. 3개 구분 규칙(STREAMING/EXACT DUPLICATES/OCR JUMP) + 이전 3개 자막 컨텍스트 |
| **Phase 3** | `core/sync_refiner.py` | 연속 프레임 ROI diff 비교. 초반 안정 구간 median baseline 대비 **2.0배 spike** = 자막 변화 시점. 등장: `start-1.0s~start`, 소멸: `end~end+1.0s` |
| **Phase 4** | `core/vlm_client.py` | 완성된 자막 목록을 LLM에 보내 최종 QC (중복 병합, 시간 교정) |

### 내보내기
| `core/subtitle_exporter.py` | SRT 또는 ASS 생성. ASS는 **BorderStyle 3(불투명 박스)** + 원래 자막 bbox 높이 기준 **동적 `\fs` 크기** |

---

## ✨ 주요 기능

### 검출 / 번역 / 동기화
- **CRAFT 검출**: `craft-text-detector` CUDA 가속, 640×360 resize
- **멀티모달 VLM**: OpenAI-compatible API. 이미지(JPEG base64) + 텍스트 프롬프트
  - 기본 모델: **`gemini-3-flash-preview:cloud`** (백업: `gemma4:31b-cloud`, 사용자 임의 모델명 입력 가능)
- **연속 프레임 diff 동기화**: 이진 탐색 대신 연속 ROI 픽셀 차이 비교 → median baseline 대비 spike로 경계 감지
- **bbox 크기 방어**: CRAFT가 세로쓰기를 full-frame으로 오감지하면(화면 20% 초과) **position 기준 ROI로 fallback**
- **짧은 자막 보장**: 0.3초 최소 duration, next_start 침범 시 50ms 간격

### 안정성
- **한글 경로 지원**: `cv2.imwrite`/`cv2.imread`를 `open()` + `cv2.imencode`/`cv2.imdecode`로 교체
- **토큰 초과 방지**: 이전 자막 컨텍스트 3개 제한, 배치 경계 정렬
- **API Pre-flight Test**: 샘플링 전 API 연결 테스트 → 실패 시 즉시 중단
- **덮어쓰기 확인**: 이미 자막이 있는 파일은 "덮어씌울까요?" 팝업
- **에러 복사**: 팝업창 텍스트 드래그/복사 가능
- **설정 저장**: API URL, 모델명, 프롬프트, 단어집을 `config.json`에 자동 저장 (API Key는 **저장 안 됨**)
- **프리셋 시스템**: URL / 모델 / 프롬프트 이름 저장·불러오기·삭제 (API Key 미포함)
- **10회 재시도 + 백업 모델 자동 전환**: 3회 실패 시 백업 모델로 전환, delay cap 12초

### GUI
- **드래그앤드롭** 파일 추가
- 여러 영상 **대기열**
- **Process Log**: 단계별 진행 상황 & 에러 출력
- **Edit & Review** 탭: `Start`, `End`, `Original`, `Translated`, `Position` 테이블에서 수동 수정
- **Glossary 단어집**: `word = translation` 형식, VLM 프롬프트 hint + 후처리 치환

---

## 🚀 설치

```bash
# 1. CUDA PyTorch (RTX 3050 기준, CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. CRAFT + 나머지 의존성
pip install craft-text-detector
pip install -r requirements.txt
```

### 모델 캐시
CRAFT 가중치는 프로젝트 폴더 **`models/`** 에 자동 저장됩니다.

### 실행
```bash
cd SubtitleVLM
python main.py
```

---

## 📋 사용법

1. **API 설정** (왼쪽 패널)
   - API Base URL: 사용하는 서비스 엔드포인트
   - API Key: 자신의 키 입력 (**저장되지 않음**)
   - Model Name: 기본 `gemini-3-flash-preview:cloud`, 또는 사용자 임의 입력
   - Backup Model: 기본 `gemma4:31b-cloud`
   - Output Format: `SRT` 또는 `ASS`
   - **Frame Interval**: CRAFT 샘플링 간격, 기본 `1.0`초

2. **API 연결 테스트** (파란 버튼)
   - "API Connection OK!" 확인 후 Start

3. **영상 추가**
   - Add File / Add Folder 버튼 또는 드래그앤드롭

4. **분석 시작**
   - "Start Analysis" 클릭
   - Phase 1(CRAFT) → Phase 2(VLM batch) → Phase 3(Sync Refiner) → Phase 4(Post-review) 순진행
   - 덮어쓰기 팝업 시 확인

5. **검수 및 저장**
   - Edit & Review 탭에서 결과 확인/수정
   - Export Subtitles 버튼으로 최종 저장

---

## ⚠️ 한계점

- **위치 추정**: CRAFT 검출 박스 기반. 좌우 동시 자막은 현재 **한 영역으로 합쳐질 수 있음**
- **색상**: VLM이 반환하는 색상은 추정 값
- **동시 자막**: 좌우 분할 자막은 하나의 ROI로 취급되어 누락 가능 (흔하지 않은 엣지 케이스)
- **CRAFT 라이브러리**: `craft-text-detector`는 유지보수 중단 상태. 내부 `np.array(polys)` 불균형 버그는 `try/except`로 회피 중

---

## 🏗️ 프로젝트 구조

```
SubtitleVLM/
├── main.py                     # PySide6 GUI + AnalysisWorker (4단계 파이프라인 오케스트레이션)
├── requirements.txt            # Python 패키지
├── config.json                 # 사용자 설정 저장 (API Key 미포함)
├── models/                     # CRAFT 가중치 캐시
└── core/
    ├── craft_filter.py         # Phase 1: CRAFT 검출, 픽셀 diff 중복 제거, dense frame 추출
    ├── vlm_client.py           # Phase 2/4: 멀티모달 VLM 배치, JSON 파서, 최종 QC
    ├── sync_refiner.py         # Phase 3: 연속 프레임 diff spike 탐색으로 등장/사라짐 교정
    ├── subtitle_exporter.py    # SRT / ASS 내보내기 (동적 크기, BorderStyle 3)
    └── ocr_extractor.py        # PaddleOCR 기반 (현재 미사용, 보존)
```

---

## 🔧 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 없음 | NVIDIA CUDA (RTX 3050 이상) |
| VRAM | - | 4GB+ (CRAFT) |
| Python | 3.10+ | 3.10 or 3.11 |
| OS | Windows 10/11 | Windows 10/11 |

---

## 📄 라이선스

MIT License
