#!/usr/bin/env python3
# Music Visualizer Bot - Оптимизированная версия для Railway

import os
import sys
import uuid
import asyncio
import logging
import shutil
import random
import math
import colorsys
import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import subprocess

import numpy as np
import librosa
import yt_dlp
from moviepy.editor import VideoClip, AudioFileClip, VideoFileClip
import ffmpeg

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# ========== НАСТРОЙКА ==========
logging.basicConfig(
    format='%(asime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Проверка токена
BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не установлен!")
    sys.exit(1)

# Конфигурация
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
MAX_DURATION = 60  # секунд
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB для Telegram

# Временные папки
TEMP_DIR = Path("/tmp/music_visualizer")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Куки из переменных окружения (опционально)
YOUTUBE_COOKIES = os.getenv('YOUTUBE_COOKIES', '')

# ========== КЛАСС ДЛЯ ЗАГРУЗКИ С ЮТУБА ==========
class YouTubeDownloader:
    """Скачивает аудио с YouTube с обходом блокировок"""
    
    def __init__(self):
        self.temp_dir = TEMP_DIR / f"yt_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(exist_ok=True)
        self.cookies_file = None
        
        # Сохраняем куки если есть
        if YOUTUBE_COOKIES:
            try:
                self.cookies_file = self.temp_dir / "cookies.txt"
                cookies_data = base64.b64decode(YOUTUBE_COOKIES).decode('utf-8')
                self.cookies_file.write_text(cookies_data)
                logger.info("✅ Куки загружены из переменных окружения")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки кук: {e}")
                self.cookies_file = None
    
    def _get_ydl_opts(self, format_type: str = 'best') -> dict:
        """Базовые опции для yt-dlp"""
        opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'ignoreerrors': True,
            'geo_bypass': True,
            
            # Эмуляция браузера
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            },
            
            # Таймауты
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            
            # Избегаем проверки сертификатов
            'nocheckcertificate': True,
        }
        
        # Добавляем куки если есть
        if self.cookies_file and self.cookies_file.exists():
            opts['cookiefile'] = str(self.cookies_file)
        
        return opts
    
    async def download_audio(self, url: str) -> Optional[Path]:
        """
        Скачивает аудио с YouTube
        Возвращает путь к mp3 файлу или None
        """
        logger.info(f"📥 Загрузка: {url}")
        
        # Пробуем разные стратегии
        strategies = [
            {'player_client': ['web']},  # Обычный веб-клиент
            {'player_client': ['android']},  # Android клиент (меньше проверок)
            {'player_client': ['ios']},  # iOS клиент
            {'player_client': ['tv']},  # TV клиент
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"🔄 Попытка {i+1}/{len(strategies)}...")
                
                opts = self._get_ydl_opts()
                opts['extractor_args'] = {
                    'youtube': {
                        'player_client': strategy['player_client'],
                        'skip': ['hls', 'dash'],
                    }
                }
                
                # Для мобильных клиентов используем меньшее качество
                if 'android' in strategy['player_client'] or 'ios' in strategy['player_client']:
                    opts['format'] = 'worstaudio/worst'
                
                loop = asyncio.get_event_loop()
                
                with yt_dlp.YoutubeDL(opts) as ydl:
                    # Скачиваем информацию
                    info = await loop.run_in_executor(
                        None, 
                        lambda: ydl.extract_info(url, download=True)
                    )
                    
                    if info is None:
                        continue
                    
                    # Ищем скачанный файл
                    audio_files = list(self.temp_dir.glob("*.mp3"))
                    if audio_files:
                        audio_path = audio_files[0]
                        logger.info(f"✅ Скачано: {audio_path.name}")
                        
                        # Проверяем размер
                        size = audio_path.stat().st_size
                        if size > MAX_FILE_SIZE:
                            logger.warning(f"⚠️ Файл слишком большой: {size/1024/1024:.1f}MB")
                            return None
                        
                        return audio_path
                    
            except Exception as e:
                logger.warning(f"❌ Стратегия {i+1} не удалась: {str(e)[:50]}")
                continue
        
        # Если ничего не сработало, пробуем через youtube-dl как fallback
        try:
            logger.info("🔄 Пробую youtube-dl...")
            import youtube_dl
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            loop = asyncio.get_event_loop()
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(
                    None, 
                    lambda: ydl.extract_info(url, download=True)
                )
                
                audio_files = list(self.temp_dir.glob("*.mp3"))
                if audio_files:
                    return audio_files[0]
                    
        except Exception as e:
            logger.error(f"❌ youtube-dl тоже не сработал: {e}")
        
        return None
    
    def cleanup(self):
        """Очистка временных файлов"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

# ========== ВИЗУАЛИЗАТОР ==========
class BeatVisualizer:
    """Создаёт визуализацию аудио"""
    
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.work_dir = TEMP_DIR / f"viz_{uuid.uuid4().hex[:8]}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем аудио
        logger.info("🎵 Анализ аудио...")
        self.y, self.sr = librosa.load(str(audio_path), duration=MAX_DURATION)
        self.duration = min(len(self.y) / self.sr, MAX_DURATION)
        
        # Анализ битов
        self.tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, 
            sr=self.sr,
            units='time'
        )
        
        # Спектральные характеристики
        self.spectral = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        logger.info(f"✅ Длительность: {self.duration:.1f}с, BPM: {self.tempo:.1f}")
    
    def get_energy(self, t: float) -> float:
        """Энергия звука в момент t"""
        idx = min(int(t * self.sr / 512), len(self.onset_env)-1)
        if idx < 0:
            return 0.1
        return float(np.clip(self.onset_env[idx] / self.onset_env.max(), 0.1, 1.0))
    
    def get_color(self, t: float) -> Tuple[int, int, int]:
        """Цвет на основе частоты"""
        idx = min(int(t * self.sr / 512), len(self.spectral)-1)
        if idx < 0:
            freq = 2000
        else:
            freq = self.spectral[idx]
        
        # Преобразуем частоту в цвет
        hue = (freq / 5000) % 1.0
        energy = self.get_energy(t)
        
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.5 + energy * 0.5)
        return (int(r*255), int(g*255), int(b*255))
    
    def make_frame(self, t: float) -> np.ndarray:
        """Создаёт кадр для момента t"""
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Текущие параметры
        color = self.get_color(t)
        energy = self.get_energy(t)
        
        # Центр экрана
        cx, cy = VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2
        
        # Радиус пульсирует в такт
        radius = int(min(VIDEO_WIDTH, VIDEO_HEIGHT) * 0.25 * (0.6 + energy * 0.8))
        
        # Создаём сетку координат
        Y, X = np.ogrid[:VIDEO_HEIGHT, :VIDEO_WIDTH]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Рисуем круг
        mask = dist <= radius
        frame[mask] = color
        
        # Рисуем кольцо если высокая энергия
        if energy > 0.3:
            ring_mask = (dist <= radius + 20) & (dist > radius)
            ring_color = tuple(min(c + 50, 255) for c in color)
            frame[ring_mask] = ring_color
        
        # Добавляем шум для эффекта
        if energy > 0.7:
            noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def create_video(self) -> Optional[Path]:
        """Создаёт видео"""
        output_path = self.work_dir / "visualizer.mp4"
        
        try:
            logger.info("🎬 Создание видео...")
            
            # Создаём видеоклип
            video_clip = VideoClip(self.make_frame, duration=self.duration)
            video_clip = video_clip.set_fps(FPS)
            
            # Добавляем аудио
            audio_clip = AudioFileClip(str(self.audio_path)).subclip(0, self.duration)
            final_clip = video_clip.set_audio(audio_clip)
            
            # Экспортируем
            final_clip.write_videofile(
                str(output_path),
                fps=FPS,
                codec='libx264',
                preset='ultrafast',  # Быстрое сжатие
                bitrate='1000k',  # Невысокий битрейт для экономии места
                audio_codec='aac',
                audio_bitrate='128k',
                threads=2,
                verbose=False,
                logger=None
            )
            
            # Закрываем клипы
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"✅ Видео готово: {size_mb:.1f}MB")
                return output_path
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания видео: {e}")
        
        return None
    
    def cleanup(self):
        """Очистка временных файлов"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

# ========== ТЕЛЕГРАМ-БОТ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "🎵 **Music Visualizer Bot**\n\n"
        "Отправь мне **ссылку на YouTube** или **аудиофайл**, "
        "и я создам визуализацию под бит!\n\n"
        f"⏱ Макс. длительность: {MAX_DURATION} сек\n"
        "⚡️ Работает даже при блокировках YouTube",
        parse_mode=ParseMode.MARKDOWN
    )

async def handle_youtube(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка YouTube ссылок"""
    url = update.message.text.strip()
    
    # Проверяем что это YouTube
    if not any(domain in url for domain in ['youtu.be', 'youtube.com', 'm.youtube.com']):
        await update.message.reply_text("❌ Пожалуйста, отправьте ссылку на YouTube")
        return
    
    # Отправляем статус
    status_msg = await update.message.reply_text(
        "⏳ **Шаг 1/3**: Подключение к YouTube...\n"
        "Это может занять до 30 секунд",
        parse_mode=ParseMode.MARKDOWN
    )
    
    downloader = None
    
    try:
        # Скачиваем аудио
        downloader = YouTubeDownloader()
        audio_path = await downloader.download_audio(url)
        
        if not audio_path:
            await status_msg.edit_text(
                "❌ Не удалось скачать видео с YouTube.\n"
                "Возможные причины:\n"
                "• Видео недоступно в вашем регионе\n"
                "• YouTube временно блокирует запросы\n"
                "• Слишком длинное видео\n\n"
                "Попробуйте другое видео или отправьте аудиофайл"
            )
            return
        
        # Проверяем размер
        file_size = audio_path.stat().st_size / 1024 / 1024
        await status_msg.edit_text(
            f"✅ **Шаг 2/3**: Аудио загружено ({file_size:.1f} MB)\n"
            "🎨 Создаю визуализацию...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Создаём визуализацию
        visualizer = BeatVisualizer(audio_path)
        video_path = visualizer.create_video()
        
        if not video_path:
            await status_msg.edit_text("❌ Ошибка при создании видео")
            return
        
        # Отправляем видео
        await status_msg.edit_text(
            "📤 **Шаг 3/3**: Отправка видео...\n"
            f"BPM: {visualizer.tempo:.1f}",
            parse_mode=ParseMode.MARKDOWN
        )
        
        with open(video_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=f"🎵 Визуализация | BPM: {visualizer.tempo:.1f}",
                supports_streaming=True,
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT
            )
        
        # Очистка
        visualizer.cleanup()
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ Произошла ошибка: {str(e)[:100]}"
        )
    finally:
        if downloader:
            downloader.cleanup()

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка аудиофайлов"""
    # Получаем файл
    audio = update.message.audio or update.message.voice
    if not audio:
        return
    
    status_msg = await update.message.reply_text(
        "⏳ **Шаг 1/2**: Загрузка аудио...",
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Определяем расширение
    if update.message.audio:
        ext = Path(audio.file_name).suffix if audio.file_name else '.mp3'
    else:
        ext = '.ogg'
    
    # Скачиваем файл
    audio_path = TEMP_DIR / f"audio_{uuid.uuid4().hex[:8]}{ext}"
    
    try:
        file = await context.bot.get_file(audio.file_id)
        await file.download_to_drive(audio_path)
        
        file_size = audio_path.stat().st_size / 1024 / 1024
        await status_msg.edit_text(
            f"✅ **Шаг 1/2**: Аудио загружено ({file_size:.1f} MB)\n"
            "🎨 **Шаг 2/2**: Создаю визуализацию...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Конвертируем в MP3 если нужно
        if audio_path.suffix != '.mp3':
            mp3_path = audio_path.with_suffix('.mp3')
            (
                ffmpeg
                .input(str(audio_path))
                .output(str(mp3_path), acodec='libmp3lame', ab='192k')
                .run(quiet=True, overwrite_output=True)
            )
            audio_path.unlink()
            audio_path = mp3_path
        
        # Создаём визуализацию
        visualizer = BeatVisualizer(audio_path)
        video_path = visualizer.create_video()
        
        if not video_path:
            await status_msg.edit_text("❌ Ошибка при создании видео")
            return
        
        # Отправляем
        with open(video_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=f"🎵 Визуализация | BPM: {visualizer.tempo:.1f}",
                supports_streaming=True
            )
        
        visualizer.cleanup()
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")
    finally:
        if audio_path.exists():
            audio_path.unlink()

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена операции"""
    await update.message.reply_text("✅ Операция отменена")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статус бота"""
    # Проверяем наличие ffmpeg
    ffmpeg_check = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
    ffmpeg_installed = ffmpeg_check.returncode == 0
    
    # Проверяем куки
    cookies_configured = bool(YOUTUBE_COOKIES)
    
    # Свободное место
    stat = shutil.disk_usage(TEMP_DIR)
    free_gb = stat.free / 1024 / 1024 / 1024
    
    status_text = (
        "📊 **Статус бота**\n\n"
        f"🍪 Куки YouTube: {'✅' if cookies_configured else '❌'}\n"
        f"🎬 FFmpeg: {'✅' if ffmpeg_installed else '❌'}\n"
        f"💾 Свободно места: {free_gb:.1f} GB\n"
        f"⏱ Макс. длительность: {MAX_DURATION} сек\n"
        f"📁 Временная папка: {TEMP_DIR}"
    )
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

# ========== ЗАПУСК ==========
def main():
    """Запуск бота"""
    # Проверяем зависимости
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("✅ FFmpeg найден")
    except:
        logger.error("❌ FFmpeg не найден! Установите ffmpeg")
        logger.info("На Railway добавьте в nixpacks.toml: apt-get install -y ffmpeg")
    
    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Добавляем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_youtube))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    
    # Запускаем
    logger.info("🚀 Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
