#!/usr/bin/env python3
# Music Visualizer Bot с поддержкой YouTube

import os
import sys
import uuid
import asyncio
import logging
import shutil
import random
import math
import colorsys
from pathlib import Path

import numpy as np
import librosa
import yt_dlp  # заменяем на yt-dlp
from moviepy.editor import VideoClip, AudioFileClip
import ffmpeg

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# ========== НАСТРОЙКА ==========
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не установлен!")
    sys.exit(1)

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
MAX_DURATION = 60
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# ========== КЛАСС ДЛЯ ЮТУБА ==========
class YouTubeDownloader:
    """Скачивает видео с YouTube с обходом блокировок"""
    
    @staticmethod
    async def download(url: str) -> Path:
        logger.info(f"📥 Скачиваю: {url}")
        
        # Настройки для обхода детекта ботов
        ydl_opts = {
            'format': 'best[height<=480]',  # 480p для быстрой загрузки
            'outtmpl': str(TEMP_DIR / f'yt_{uuid.uuid4().hex}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],  # эмуляция клиентов
                    'skip': ['hls', 'dash'],
                }
            },
            # Имитация браузера
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }
        
        try:
            loop = asyncio.get_event_loop()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Извлекаем информацию и скачиваем
                info = await loop.run_in_executor(None, ydl.extract_info, url, True)
                filename = ydl.prepare_filename(info)
                
                # Ищем файл (может быть .mp4 или другой)
                path = Path(filename)
                if not path.exists():
                    # Пробуем найти любой видеофайл
                    files = list(TEMP_DIR.glob(f"*{path.stem}*"))
                    if files:
                        path = files[0]
                
                logger.info(f"✅ Скачано: {path.name}")
                return path
                
        except Exception as e:
            logger.error(f"❌ Ошибка YouTube: {e}")
            raise

# ========== ВИЗУАЛИЗАТОР ==========
class BeatVisualizer:
    """Создаёт визуализацию аудио"""
    
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.work_dir = TEMP_DIR / f"viz_{uuid.uuid4().hex}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Анализ аудио
        self.y, self.sr = librosa.load(str(audio_path))
        self.duration = len(self.y) / self.sr
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        self.spectral = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        
        logger.info(f"🎵 Аудио: {self.duration:.1f} сек, BPM: {self.tempo:.1f}")
    
    def get_beat_energy(self, t: float) -> float:
        """Энергия бита (0-1)"""
        if len(self.beat_times) == 0:
            return 0.5
        closest = min(self.beat_times, key=lambda x: abs(x - t))
        distance = abs(closest - t)
        return math.exp(-distance * 30) if distance < 0.2 else 0.1
    
    def get_color(self, t: float) -> tuple:
        """Цвет на основе частоты"""
        idx = min(int(t * self.sr / 512), len(self.spectral)-1)
        freq = self.spectral[idx] if idx >= 0 else 2000
        hue = (freq / 5000) % 1.0
        energy = self.get_beat_energy(t)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.3 + energy * 0.7)
        return (int(r*255), int(g*255), int(b*255))
    
    def make_frame(self, t: float) -> np.array:
        """Создаёт кадр"""
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Фон
        bg_color = self.get_color(t - 0.1)
        frame[:, :] = [c // 4 for c in bg_color]
        
        # Центральный круг
        color = self.get_color(t)
        energy = self.get_beat_energy(t)
        radius = int(min(VIDEO_WIDTH, VIDEO_HEIGHT) * (0.2 + energy * 0.8) * 0.25)
        center_x, center_y = VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2
        
        Y, X = np.ogrid[:VIDEO_HEIGHT, :VIDEO_WIDTH]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        frame[dist <= radius] = color
        
        # Кольцо
        if energy > 0.4:
            ring = (dist <= radius * 1.3) & (dist > radius)
            ring_color = tuple(min(c + 30, 255) for c in color)
            frame[ring] = ring_color
        
        return frame
    
    def create_video(self) -> Path:
        """Создаёт видео из аудио"""
        output_path = self.work_dir / "visualizer.mp4"
        
        video_duration = min(self.duration, MAX_DURATION)
        video_clip = VideoClip(self.make_frame, duration=video_duration)
        video_clip = video_clip.set_fps(FPS)
        
        audio_clip = AudioFileClip(str(self.audio_path)).subclip(0, video_duration)
        final_clip = video_clip.set_audio(audio_clip)
        
        final_clip.write_videofile(
            str(output_path),
            fps=FPS,
            codec='libx264',
            preset='medium',
            bitrate='2000k',
            audio_codec='aac',
            audio_bitrate='192k',
            threads=4,
            verbose=False,
            logger=None
        )
        
        return output_path
    
    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

# ========== ТЕЛЕГРАМ-БОТ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎵 **Music Visualizer Bot**\n\n"
        "Отправь **ссылку на YouTube** или **аудиофайл**, "
        "а я сделаю клип с визуализацией под бит!\n\n"
        f"⏱ Макс длительность: {MAX_DURATION} сек",
        parse_mode=ParseMode.MARKDOWN
    )

async def handle_youtube(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ссылок YouTube"""
    url = update.message.text.strip()
    
    if not ('youtu.be' in url or 'youtube.com' in url):
        await update.message.reply_text("❌ Это не ссылка YouTube")
        return
    
    status = await update.message.reply_text("⏳ Скачиваю видео с YouTube...")
    
    try:
        # Скачиваем видео
        video_path = await YouTubeDownloader.download(url)
        
        # Извлекаем аудио
        audio_path = TEMP_DIR / f"audio_{uuid.uuid4().hex}.mp3"
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path))
            .run(quiet=True, overwrite_output=True)
        )
        
        # Создаём визуализацию
        await status.edit_text("🎨 Создаю визуализацию под бит...")
        visualizer = BeatVisualizer(audio_path)
        result_path = visualizer.create_video()
        
        # Отправляем
        await status.edit_text("📤 Отправляю...")
        with open(result_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=f"🎵 Клип под бит | BPM: {visualizer.tempo:.1f}",
                supports_streaming=True
            )
        
        # Очистка
        visualizer.cleanup()
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        await status.delete()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await status.edit_text(f"❌ Ошибка: {str(e)[:100]}\nYouTube мог заблокировать запрос.")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка аудиофайлов"""
    audio = update.message.audio or update.message.voice
    if not audio:
        return
    
    status = await update.message.reply_text("⏳ Обрабатываю аудио...")
    
    ext = '.mp3' if update.message.audio else '.ogg'
    audio_path = TEMP_DIR / f"audio_{uuid.uuid4().hex}{ext}"
    
    file = await context.bot.get_file(audio.file_id)
    await file.download_to_drive(audio_path)
    
    try:
        visualizer = BeatVisualizer(audio_path)
        result_path = visualizer.create_video()
        
        with open(result_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=f"🎵 Клип под бит | BPM: {visualizer.tempo:.1f}",
                supports_streaming=True
            )
        
        visualizer.cleanup()
        await status.delete()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await status.edit_text(f"❌ Ошибка: {e}")
    finally:
        audio_path.unlink(missing_ok=True)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Отменено")

async def post_init(app: Application):
    me = await app.bot.get_me()
    print(f"\n🤖 Бот: @{me.username}")
    print(f"📁 Временная папка: {TEMP_DIR}")

# ========== ЗАПУСК ==========
def main():
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_youtube))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    
    print("\n🎵 MUSIC VISUALIZER BOT")
    print("="*50)
    app.run_polling()

if __name__ == '__main__':
    main()
