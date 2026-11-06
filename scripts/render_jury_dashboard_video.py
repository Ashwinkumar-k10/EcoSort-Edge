"""
scripts/render_jury_dashboard_video.py
Renders the complete synchronized jury showcase video combining:
1. assets/dashboard_canvas.png (1920x1080 UI canvas)
2. R1_Screen_recording.mp4 (scaled to 1200x640 at x=25, y=97)
3. camara_rec.mp4 (scaled to 640x360 at x=1255, y=97)
4. Muted audio (zero noise, purely professional)
5. Animated bottom subtitles in crisp white text across the 80-second duration via libass.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANVAS = PROJECT_ROOT / "assets" / "dashboard_canvas.png"
SCREEN_VID = PROJECT_ROOT / "R1_Screen_recording.mp4"
CAM_VID = PROJECT_ROOT / "camara_rec.mp4"
SUBTITLE_FILE = PROJECT_ROOT / "assets" / "subtitles.srt"
OUTPUT_VID = PROJECT_ROOT / "assets" / "EcoSort_Edge_Jury_Showcase.mp4"

def render_video():
    print("[*] Preparing FFmpeg filtergraph for Option 3 Dashboard Showcase...")

    sub_path_escaped = str(SUBTITLE_FILE).replace("\\", "/").replace(":", "\\:")

    filter_complex = (
        f"[1:v]scale=1200:640:flags=lanczos[scr];"
        f"[2:v]scale=640:360:flags=lanczos[cam];"
        f"[0:v][scr]overlay=25:97[bg1];"
        f"[bg1][cam]overlay=1255:97[bg2];"
        f"[bg2]subtitles='{sub_path_escaped}':force_style='FontSize=16,FontName=Segoe UI,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Alignment=2,MarginV=26'[outv]"
    )

    cmd = [
        "ffmpeg",
        "-loop", "1", "-i", str(CANVAS),
        "-i", str(SCREEN_VID),
        "-i", str(CAM_VID),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", "79.6",
        "-an",
        "-y",
        str(OUTPUT_VID)
    ]

    print("[*] Running FFmpeg rendering command (this takes ~15-20 seconds)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] FFmpeg error:")
        print(result.stderr[-2000:])
        sys.exit(1)

    print(f"[+] Final Jury Showcase Video Successfully Rendered: {OUTPUT_VID}")
    print(f"    Size: {OUTPUT_VID.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    render_video()
