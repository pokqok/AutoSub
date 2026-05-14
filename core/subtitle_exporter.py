import datetime
from typing import List, Dict

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
        # centiseconds (0~99)
        cents = int(td.microseconds / 10000)
        return f"{hours:01}:{minutes:02}:{secs:02}.{cents:02}"

    def generate_srt(self, subtitles: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(subtitles, 1):
                f.write(f"{idx}\n{self.format_time_srt(sub['start'])} --> {self.format_time_srt(sub['end'])}\n{sub['translated']}\n\n")

    def generate_ass(self, subtitles: List[Dict], output_path: str):
        """
        추출된 색상 정보를 바탕으로 ASS 스타일 태그가 적용된 파일을 생성합니다.
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
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        def rgb_to_ass_color(hex_color: str) -> str:
            if not hex_color or not hex_color.startswith('#') or len(hex_color) != 7:
                return "&H00FFFFFF&" 
            r, g, b = hex_color[1:3], hex_color[3:5], hex_color[5:7]
            return f"&H{b}{g}{r}&"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for sub in subtitles:
                start = self.format_time_ass(sub['start'])
                end = self.format_time_ass(sub['end'])
                color_tag = rgb_to_ass_color(sub.get('color', '#FFFFFF'))
                text = f"{{\\c{color_tag}}}{sub['translated']}"
                # Effect 필드는 비워두고 공백 없이 출력
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
