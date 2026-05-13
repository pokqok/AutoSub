import datetime
from typing import List, Tuple

class SubtitleExporter:
    """
    분석 결과를 바탕으로 SRT 또는 ASS 자막 파일을 생성하는 클래스
    """
    @staticmethod
    def format_time_srt(seconds: float) -> str:
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    @staticmethod
    def format_time_ass(seconds: float) -> str:
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        cents = int((td.microseconds / 1000) * 10) # ASS는 1/100초 단위
        return f"{hours:01}:{minutes:02}:{secs:02}.{cents:02}"

    def generate_srt(self, subtitles: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(subtitles, 1):
                f.write(f"{idx}\n{self.format_time_srt(sub['start'])} --> {self.format_time_srt(sub['end'])}\n{sub['text']}\n\n")

    def generate_ass(self, subtitles: List[Dict], output_path: str):
        """
        기본 스타일의 ASS 파일을 생성합니다.
        """
        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "ScaledCropsHeight: 0",
            "ScaledCropsWidth: 0",
            "WrapStyle: 0",
            "Encoding: UTF-8",
            "",
            "[V4+ Styles]",
            "Format: Name, Font, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, EdgeMode, Blur, Alpha",
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,1,0,2",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for sub in subtitles:
                start = self.format_time_ass(sub['start'])
                end = self.format_time_ass(sub['end'])
                # ASS 형식: Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,effetto, {sub['text']}\n")
