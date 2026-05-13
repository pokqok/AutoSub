# SubtitleVLM - AI Powered Hardcoded Subtitle Extractor

This program extracts hardcoded Japanese subtitles from videos, translates them to Korean, and generates an SRT file using state-of-the-art VLMs (Vision Language Models).

## Features
- **VLM Integration**: Works with DeepSeek-V de Pro, GLM-5.1, Gemma 4, etc.
- **Smart Frame Sampling**: Detects subtitle changes using Canny Edge detection to minimize API calls.
- **Dialogue Filtering**: Uses LLM's visual reasoning to distinguish between real dialogue and background text/SFX.
- **Precise Timing**: Merges identical consecutive texts to create natural subtitle blocks.
- **User-Friendly GUI**: Manage API keys and file paths easily.

## Installation
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. Enter your VLM API Key and select the model.
2. Select the input video file and the output destination folder.
3. Click "Start Extraction".
4. The program will sample frames $\rightarrow$ analyze via VLM $\rightarrow$ generate `.srt` file.
