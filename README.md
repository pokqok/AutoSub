# AutoSub - AI 하드코딩 자막 추출·번역기

CRAFT 검출 + VLM 멀티모달 배치 + 이진 탐색 동기화 파이프라인으로, 영상 속 **하드코딩 일본어 자막**을 검출해 **한국어 자막(SRT / ASS)**으로 변환하는 Windows GUI 애플리케이션입니다.

## 🎯 핵심 개념

**"CRAFT가 자막이 나오는 프레임을 뽑고 → VLM이 이미지로 읽어 한꺼번에 번역 → 이진 탐색으로 정확한 등장/사라짐 교정"**

### 3단계 파이프라인

| 단계 | 파일 | 역할 |
|------|------|------|
| **Phase 1** | `core/craft_filter.py` | 영상에서 1초 간격으로 프레임을 샘플링 → CRAFT로 자막 영역 검출 → 히스토그램 기반 중복 제거 + ROI 크롭 → `timestamp`, `bbox`, `filepath` 메타데이터 전달 |
| **Phase 2** | `core/vlm_client.py` | 최대 10장씩 배치로 멀티모달 VLM에 전달. 일본어 원문 → 한국어 번역, 색상, 위치, 타이밍 추정. 이전 3개 자막 컨텍스트 전달로 배치 간 중복 방지 |
| **Phase 3** | `core/sync_refiner.py` | 이진 탐색으로 0.1초 단위 **등장/사라짐 경계**를 정밀 추적. Edge 형태 + NCC 구조적 유사도로 배경 색 변화와 무관하게 **같은 자막 vs 다른 자막**을 구분 |

### 내보내기
|`core/subtitle_exporter.py` | SRT(default) 또는 ASS(색상 태그 `\c&HBBGGRR&`, font size 18) 생성 |

---

## ✨ 주요 기능

### 검출 / 번역 / 동기화
- **CRAFT 검출**: `craft-text-detector` 기반, CUDA 가속, 640×360 resize로 GPU 빠른 추론
- **멀티모달 VLM**: OpenAI-compatible `/v1/chat/completions` API. 이미지(JPEG base64) + 텍스트 프롬프트
  - 기본 모델: **`gemini-3-flash-preview:cloud`** (대체: `gemma3:cloud`, 사용자 임의 모델명 입력 가능)
- **이진 탐색 동기화 (0.1초)**: 자막 참조 ROI와 비교하며 "없음→있음"(등장) / "있음→없음"(사라짐) 경계를 좁혀감
- **배경 색 불변 유사도**: Edge(Canny) + NCC(정규화 상호상관) 6:4 혼합 → 배경이 바뀌어도 같은 자막/다른 자막 구분
- **단위 자막 보장**: 짧은 대사(max 0.8s), 전체 자막(max 3.0s), 자막 간 0.05s 간격 유지

### 안정성
- **토큰 초과 방지**: 이전 자막 컨텍스트를 10개 → **3개**로 제한, 배치 장렬 누락 방지
- **API Pre-flight Test**: 샘플링 전 API 연결 테스트 → 실패 시 즉시 중단
- **덮어쓰기 확인**: 이미 자막이 있는 파일은 "덮어씌울까요?" 팝업
- **에러 복사**: 팝업창 텍스트 드래그/복사 가능
- **설정 저장**: API URL, 모델명, 프롬프트, 단어집을 `config.json`에 자동 저장 (API Key는 **저장 안 됨**)
- **프리셋 시스템**: URL / 모델 / 프롬프트를 이름 저장·불러오기·삭제 (API Key 미포함)

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
pip install craft-text-detector -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 모델 캐시
CRAFT 가중치는 프로젝트 폴더 **`models/`** 에 자동 저장됩니다 (C드라이브 공간 문제 방지).

### 실행
```bash
cd SubtitleVLM
python main.py
```

---

## 📋 사용법

1. **API 설정** (왼쪽 패널)
   - API Base URL: 사용하는 서비스 엔드포인트 (예: `https://api.openai.com/v1`)
   - API Key: 자신의 키 입력 (**저장되지 않음**)
   - Model Name: 기본 `gemini-3-flash-preview:cloud`, 또는 사용자 임의 입력
   - Output Format: `SRT` 또는 `ASS`
   - OCR Interval: CRAFT 샘플링 간격, 기본 `1.0`초

2. **API 연결 테스트** (파란 버튼)
   - "API Connection OK!" 확인 후 Start

3. **영상 추가**
   - Add File / Add Folder 버튼 또는 드래그앤드롭

4. **분석 시작**
   - "Start Analysis" 클릭
   - Phase 1(CRAFT) → Phase 2(VLM batch) → Phase 3(Sync Refiner) 순진행
   - 덮어쓰기 팭업 시 확인

5. **검수 및 저장**
   - Edit & Review 탭에서 결과 확인/수정
   - Export Subtitles 버튼으로 최종 저장

---

## ⚠️ 한계점

- **위치 추정**: CRAFT 검출 박스 기반. 왼쪽위+오른쪽위에 **동시에** 자막이 뜨는 경우, 현재는 **한 영역으로 합쳐져** 전달됨. 향후 개별 bbox 분리 개선 예정
- **색상**: VLM이 반환하는 색상은 추정 값이며, 영상 필터/톤에 따라 달라질 수 있음
- **동시 자막**: 좌우 분할 자막은 현재 하나의 ROI로 취급되어 누락 가능 (흔하진 않은 엣지 케이스)
- **연속 대사**: 같은 위치에서 **줄임없이** 이어지는 자막(0.1초 이내 겹침)은 구분 한계가 있음
- **CRAFT 라이브러리**: `craft-text-detector`는 더 이상 유지보수 중단. 호환성 이슈(불균일 배열 예외 등)는 `try/except`로 회피 중

---

## 🏗️ 프로젝트 구조

```
SubtitleVLM/
├── main.py                     # PySide6 GUI + AnalysisWorker (3단계 파이프라인 오케스트레이션)
├── requirements.txt            # Python 패키지
├── config.json                 # 사용자 설정 저장 (API Key 미포함)
├── models/                     # torch hub / CRAFT 가중치 캐시
└── core/
    ├── craft_filter.py         # Phase 1: CRAFT 검출, ROI 크롭, 히스토그램 중복 제거, bbox 메타데이터
    ├── vlm_client.py           # Phase 2: 멀티모달 VLM 배치, 이전 자막 컨텍스트, JSON 파서
    ├── sync_refiner.py         # Phase 3: 이진 탐색 + Edge/NCC 유사도로 등장/사라짐 교정
    ├── subtitle_exporter.py     # SRT / ASS 내보내기 (색상 태그 지원)
    └── ocr_extractor.py        # PaddleOCR 기반 SubtitleFrameFilter (현재 미사용, 보존)
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
