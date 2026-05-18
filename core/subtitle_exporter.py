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
        cents = int(td.microseconds / 10000)
        return f"{hours:01}:{minutes:02}:{secs:02}.{cents:02}"

    @staticmethod
    def _calc_fontsize(sub: Dict, playresy: int = 1080) -> int:
        """원래 자막 bbox 높이를 기반으로 ASS \\fs 태그 크기 계산."""
        bbox = sub.get('bbox')
        orig_h = sub.get('orig_h', 1080)
        if bbox and len(bbox) == 4:
            h = bbox[3] - bbox[1]
            scale = playresy / orig_h
            fontsize = int(h * scale * 1.0)
            return max(18, min(72, fontsize))
        return 24

    def generate_srt(self, subtitles: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(subtitles, 1):
                text = sub['translated']
                f.write(
                    f"{idx}\n"
                    f"{self.format_time_srt(sub['start'])} --> {self.format_time_srt(sub['end'])}\n"
                    f"{text}\n\n"
                )

    def generate_ass(self, subtitles: List[Dict], output_path: str):
        """ASS 생성. 원래 자막 크기(\\fs) 태그 적용."""
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
            "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,1,0,2,10,10,10,1",
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
                fontsize = self._calc_fontsize(sub)
                text = f"{{\\fs{fontsize}\\c{color_tag}}}{sub['translated']}"
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
