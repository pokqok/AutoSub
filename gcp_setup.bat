@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ══════════════════════════════════════════
::  GCP Vertex AI 원클릭 자동 설정
::  사용자 개입: 브라우저 로그인 1회만
:: ══════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════╗
echo  ║  GCP Vertex AI 원클릭 설정 시작     ║
echo  ╚══════════════════════════════════════╝
echo.

:: ── gcloud 확인 ──
where gcloud >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] gcloud CLI 미설치
    echo  https://cloud.google.com/sdk/docs/install
    echo  설치 후 다시 실행하세요.
    pause
    exit /b 1
)

:: ── 자동 변수 설정 ──
:: 타임스탬프 + 랜덤으로 고유 ID 생성 (충돌 방지)
for /f %%a in ('powershell -Command "Get-Date -Format yyyyMMddHHmmss"') do set "TS=%%a"
set /a R1=%random% %% 9000 + 1000
set "PROJECT_ID=vtx-%TS%-%R1%"
set "SA_NAME=vertex-sa"
set "KEY_FILE=%~dp0vertex-key-%TS%.json"

:: ══════════════════════════════════════════
echo  [1/8] 로그인 중... 브라우저에서 계정을 선택하세요.
echo.
:: 기존 세션 충돌 방지: 새 계정으로 강제 로그인
call gcloud auth revoke --all --quiet >nul 2>&1
call gcloud auth login --quiet --force 2>nul
if %errorlevel% neq 0 (
    echo [오류] 로그인 실패. 다시 실행하세요.
    pause
    exit /b 1
)

:: 로그인된 계정 확인
for /f "delims=" %%a in ('gcloud config get-value account 2^>nul') do set "ACCOUNT=%%a"
echo  [OK] %ACCOUNT%
echo.

:: ══════════════════════════════════════════
echo  [2/8] 결제 계정 자동 탐색...

:: 열린 결제 계정 중 첫 번째 자동 선택
set "BILLING_ID="
for /f "delims=" %%a in ('gcloud billing accounts list --filter="open=true" --format="value(name)" --limit=1 2^>nul') do set "BILLING_ID=%%a"

if not defined BILLING_ID (
    echo [오류] 활성 결제 계정 없음!
    echo.
    echo  ▶ 해결: 아래 링크를 브라우저에서 열어 무료체험을 활성화하세요.
    echo    https://console.cloud.google.com/freetrial
    echo.
    echo  활성화 후 이 스크립트를 다시 실행하세요.
    pause
    exit /b 1
)
echo  [OK] 결제 계정: %BILLING_ID%
echo.

:: ══════════════════════════════════════════
echo  [3/8] 프로젝트 생성: %PROJECT_ID%
call gcloud projects create %PROJECT_ID% --name="Vertex-Auto" --quiet 2>&1 | findstr /v "WARNING"
if %errorlevel% neq 0 (
    :: 한번 더 시도 (ID 충돌 대비)
    set /a R2=%random% %% 9000 + 1000
    set "PROJECT_ID=vtx-%TS%-%R2%"
    call gcloud projects create !PROJECT_ID! --name="Vertex-Auto" --quiet 2>nul
    if !errorlevel! neq 0 (
        echo [오류] 프로젝트 생성 실패. 프로젝트 한도 초과일 수 있음.
        pause
        exit /b 1
    )
)
call gcloud config set project %PROJECT_ID% --quiet >nul 2>&1
echo  [OK] 프로젝트 생성 완료
echo.

:: ══════════════════════════════════════════
echo  [4/8] 결제 계정 연결...
call gcloud billing projects link %PROJECT_ID% --billing-account=%BILLING_ID% --quiet 2>&1 | findstr /v "WARNING"
if %errorlevel% neq 0 (
    echo [오류] 결제 연결 실패
    pause
    exit /b 1
)
echo  [OK] 결제 연결 완료
echo.

:: ══════════════════════════════════════════
echo  [5/8] API 활성화 중... (1~3분 소요, 기다려주세요)
call gcloud services enable aiplatform.googleapis.com --project=%PROJECT_ID% --quiet 2>nul
if %errorlevel% neq 0 (
    echo [오류] Vertex AI API 활성화 실패
    pause
    exit /b 1
)
call gcloud services enable iam.googleapis.com --project=%PROJECT_ID% --quiet 2>nul
call gcloud services enable cloudresourcemanager.googleapis.com --project=%PROJECT_ID% --quiet 2>nul
echo  [OK] API 활성화 완료
echo.

:: ══════════════════════════════════════════
echo  [6/8] 서비스 계정 생성...
set "SA_EMAIL=%SA_NAME%@%PROJECT_ID%.iam.gserviceaccount.com"
call gcloud iam service-accounts create %SA_NAME% --display-name="Vertex-SA" --project=%PROJECT_ID% --quiet 2>nul
if %errorlevel% neq 0 (
    echo [오류] 서비스 계정 생성 실패
    pause
    exit /b 1
)
echo  [OK] %SA_EMAIL%
echo.

:: ══════════════════════════════════════════
echo  [7/8] 권한 부여...
call gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SA_EMAIL%" --role="roles/aiplatform.user" --quiet >nul 2>&1
call gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SA_EMAIL%" --role="roles/aiplatform.admin" --quiet >nul 2>&1
echo  [OK] 권한 부여 완료
echo.

:: ══════════════════════════════════════════
echo  [8/8] JSON 키 생성...
call gcloud iam service-accounts keys create "%KEY_FILE%" --iam-account=%SA_EMAIL% --project=%PROJECT_ID% --quiet 2>nul
if %errorlevel% neq 0 (
    echo [오류] 키 생성 실패
    pause
    exit /b 1
)
echo  [OK] 키 저장됨
echo.

:: ══════════════════════════════════════════
:: 환경변수 자동 설정 (현재 세션)
set "GOOGLE_APPLICATION_CREDENTIALS=%KEY_FILE%"

echo.
echo  ╔══════════════════════════════════════╗
echo  ║            완료!                     ║
echo  ╚══════════════════════════════════════╝
echo.
echo   계정     : %ACCOUNT%
echo   프로젝트 : %PROJECT_ID%
echo   키 파일  : %KEY_FILE%
echo.
echo   이 세션에서 GOOGLE_APPLICATION_CREDENTIALS 자동 설정됨
echo.

pause
