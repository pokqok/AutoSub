# AutoSub - AI Japanese Subtitle Extractor

OCR + LLM 하이브리드 파이프라인으로 일본어 하드코딩 자막이 박힌 영상을 한국어 자막(SRT/ASS)으로 변환하는 Windows GUI 애플리케이션입니다.

## 🎯 핵심 개념

**"OCR이 원문+시간을 대충 뽑고 → LLM이 번역+정제를 배치로 처리한다"**

1. **Phase 1 (OCR)**: PaddleOCR 또는 PaddleOCR-VL-1.5로 영상 전체에서 일본어 원문과 시간을 추출
2. **Phase 2 (LLM)**: OpenAI/호환 API를 통해 텍스트 LLM이 한 번에 정제+번역
   - OCR 잘못 분할 → 하나로 병합
   - 배경 노이즈(UI, 로고, 제작사명) → 제거
   - 톤/상황 반영 자연스러운 한국어 번역
   - 대화 호흡에 맞는 시간 교정
3. **Phase 3 (Export)**: SRT 또는 ASS로 내보내기

## ✨ 주요 기능

### OCR 엔진 선택
| 엔진 | 특징 | 권장 상황 |
|------|------|----------|
| **PaddleOCR** | 빠르고 가벼움 (로컬 CPU/GPU) | 일반적인 폰트, 세로쓰기 있는 영상 |
| **PaddleOCR-VL-1.5** | 문서 특화 VLM (로컬 GPU) | 왜곡된 화면, 스크린샷, 스캔 영상 |

- RTX 3050 8GB 이상 권장 (VL-1.5: 약 3.5~5GB VRAM 사용)

### 안정성
- **API Pre-flight Test**: 샘플링 전 API 연결 테스트 → 실패하면 즉시 중단 (시간 낭비 방지)
- **덮어쓰기 확인**: 이미 자막 있는 파일은 "덮어씌울까요?" 팝업
- **에러 복사**: 팝업창 텍스트 드래그/복사 가능
- **자막 설정 저장**: API Key, URL, 프롬프트, 단어집 등 `config.json` 자동 저장

### GUI
- 드래그앤드롭 파일 추가
- 여러 영상 큐 대기열
- Edit & Review 탭: OCR 원문 ↔ 번역문 비교 및 수동 수정
- Glossary 단어집 관리 (VLM 프롬프트 힌트 + 후처리 치환)

## 🚀 설치

### Python 의존성
```bash
pip install -r requirements.txt
```

### PaddleOCR GPU 가속 (선택)
**PaddleOCR-VL-1.5 사용 시 반드시 필요:**

```bash
# CUDA 12.6 (Windows/Linux)
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# CPU only (PaddleOCR 사용 시에도 가능)
pip install paddlepaddle
```

**참고:** PaddleOCR-VL-1.5는 첫 실행 시 HuggingFace로부터 약 1.8GB 모델을 자동 다운로드합니다 (캐시 저장, 한 번만).

### 실행
```bash
cd SubtitleVLM
python main.py
```

## 📋 사용법

1. **API 설정** (왼쪽 패널)
   - API Base URL: `https://api.openai.com` 또는 사용하는 서비스
   - API Key: 자신의 키 입력
   - Model Name: `gpt-4o`, `gemini-2.5-flash`, `kimi-k2.6` 등 (텍스트 LLM)
   - OCR Engine: `PaddleOCR` 또는 `PaddleOCR-VL-1.5` 선택
   - OCR Interval: `0.3`초 (기본) / `0.5` / `1.0`
   - Output Format: `SRT` 또는 `ASS`

2. **API 연결 테스트** (파란 버튼)
   - "API Connection OK!" 확인 후 Start

3. **영상 추가**
   - Add File / Add Folder 버튼 또는 드래그앤드롭

4. **분석 시작**
   - "Start Analysis" 클릭
   - 덮어쓰기 팝업 확인 후 진행

5. **검수 및 저장**
   - Edit & Review 탭에서 결과 확인/수정
   - Export Subtitles 버튼으로 최종 저장

## ⚠️ 한계점

- **OCR 인식률**: PaddleOCR/VL-1.5 자체의 인식 한계 존재 (특이한 폰트, 빠른 전환)
- **색상**: 현재 OCR 기반으로는 색상 정보를 얻지 못함 (ASS 색상 태그는 기본 흰색)
- **위치**: PaddleOCR은 대략적 위치 반환, VL-1.5은 위치 없음 (PaddleOCR 사용 권장)
- **타임라인 교정**: LLM이 "자연스럽게" 교정하지만 완벽하지는 않음 (수동 검수 필요)
- **GPU VRAM**: PaddleOCR-VL-1.5는 최소 4GB VRAM 권장 (bf16 기준)

## 🏗️ 프로젝트 구조

```
SubtitleVLM/
├── main.py                # PySide6 GUI 메인
├── requirements.txt       # Python 패키지 의존성
├── config.json            # 사용자 설정 저장
└── core/
    ├── ocr_extractor.py       # PaddleOCR (로컬, 가벼움)
    ├── paddle_vl_extractor.py # PaddleOCR-VL-1.5 (로컬 GPU)
    ├── llm_client.py          # OpenAI 호환 텍스트 LLM 클라이언트
    └── subtitle_exporter.py   # SRT / ASS 생성
```

## 📄 라이선스

MIT License
