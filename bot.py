#!/usr/bin/env python3
# ███╗   ███╗██╗   ██╗███████╗██╗ ██████╗    ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗     ██╗███████╗██████╗ 
# ████╗ ████║██║   ██║██╔════╝██║██╔════╝    ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║     ██║╚══███╔╝██╔══██╗
# ██╔████╔██║██║   ██║███████╗██║██║         ██║   ██║██║███████╗██║   ██║███████║██║     ██║  ███╔╝ ██████╔╝
# ██║╚██╔╝██║██║   ██║╚════██║██║██║         ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══██╗
# ██║ ╚═╝ ██║╚██████╔╝███████║██║╚██████╗     ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗██║███████╗██║  ██║
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝      ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚═╝  ╚═╝
#                              AUDIO → BEAT VISUALIZER
#                 Отправляешь аудио → получаешь клип с визуализацией под бит

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
from datetime import datetime

import numpy as np
import librosa
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip
from moviepy.video.VideoClip import ColorClip
import ffmpeg

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.constants import ParseMode

# ========== НАСТРОЙКА ==========
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не установлен!")
    sys.exit(1)

# Параметры видео
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
MAX_DURATION = 60  # Максимальная длина видео (сек)
MAX_FILE_SIZE = 45 * 1024 * 1024  # Telegram лимит ~50 MB, оставляем запас

# Временные директории
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# ========== ВИЗУАЛИЗАТОР ==========
class BeatVisualizer:
    """
    Создаёт визуализацию аудио с реакцией на биты
    """
    
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.work_dir = TEMP_DIR / f"viz_{uuid.uuid4().hex}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем аудио и анализируем
        self.y, self.sr = librosa.load(str(audio_path))
        self.duration = len(self.y) / self.sr
        
        # Анализируем биты
        self.tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, 
            sr=self.sr,
            units='time'
        )
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        
        # Анализируем спектр для цветов
        self.spectral = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        
        logger.info(f"🎵 Аудио: {self.duration:.1f} сек, BPM: {self.tempo:.1f}, битов: {len(self.beat_times)}")
    
    def get_beat_energy(self, t: float) -> float:
        """
        Возвращает энергию бита в момент времени t (0-1)
        """
        # Находим ближайший бит
        if len(self.beat_times) == 0:
            return 0.5
        
        # Расстояние до ближайшего бита
        closest = min(self.beat_times, key=lambda x: abs(x - t))
        distance = abs(closest - t)
        
        # Энергия затухает после бита (гауссиан)
        if distance < 0.1:
            return math.exp(-distance * 50)  # резкий пик
        return 0.2
    
    def get_color(self, t: float) -> tuple:
        """
        Возвращает цвет на основе частоты в момент t
        """
        # Определяем индекс времени в спектре
        idx = min(int(t * self.sr / 512), len(self.spectral)-1)
        if idx < 0:
            idx = 0
        
        # Частота влияет на оттенок (hue)
        freq = self.spectral[idx]
        hue = (freq / 5000) % 1.0  # нормализуем
        
        # Энергия бита влияет на яркость
        energy = self.get_beat_energy(t)
        
        # Конвертируем HSV в RGB
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.5 + energy * 0.5)
        return (int(r*255), int(g*255), int(b*255))
    
    def get_size(self, t: float) -> float:
        """
        Возвращает размер круга/элемента в момент t
        """
        energy = self.get_beat_energy(t)
        return 0.3 + energy * 0.7  # от 30% до 100% экрана
    
    def make_frame(self, t: float) -> np.array:
        """
        Создаёт один кадр визуализации для момента t
        """
        # Создаём пустой кадр
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Получаем цвет и размер
        color = self.get_color(t)
        size_factor = self.get_size(t)
        
        # Рисуем центральный круг
        center_x = VIDEO_WIDTH // 2
        center_y = VIDEO_HEIGHT // 2
        radius = int(min(VIDEO_WIDTH, VIDEO_HEIGHT) * size_factor * 0.3)
        
        # Векторизованное рисование круга (быстрее чем цикл)
        Y, X = np.ogrid[:VIDEO_HEIGHT, :VIDEO_WIDTH]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        
        frame[mask] = color
        
        # Добавляем внешнее кольцо если есть энергия
        energy = self.get_beat_energy(t)
        if energy > 0.6:
            ring_radius = int(radius * 1.5)
            ring_mask = (dist_from_center <= ring_radius) & (~mask)
            # Более светлый цвет для кольца
            ring_color = tuple(min(c + 50, 255) for c in color)
            frame[ring_mask] = ring_color
        
        # Добавляем случайные частицы на сильных битах
        if energy > 0.8:
            num_particles = int(energy * 20)
            for _ in range(num_particles):
                px = random.randint(0, VIDEO_WIDTH-1)
                py = random.randint(0, VIDEO_HEIGHT-1)
                particle_color = tuple(min(c + 100, 255) for c in color)
                frame[py, px] = particle_color
        
        return frame
    
    def create_video(self) -> Path:
        """
        Создаёт видео из аудио с визуализацией
        """
        output_path = self.work_dir / "visualizer.mp4"
        
        # Создаём видеоклип из функции make_frame
        video_clip = VideoClip(self.make_frame, duration=min(self.duration, MAX_DURATION))
        video_clip = video_clip.set_fps(FPS)
        
        # Добавляем аудио
        audio_clip = AudioFileClip(str(self.audio_path)).subclip(0, min(self.duration, MAX_DURATION))
        final_clip = video_clip.set_audio(audio_clip)
        
        # Экспортируем с оптимизацией для Telegram
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
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ Видео создано: {output_path}, размер: {file_size/1024/1024:.1f} MB")
        
        # Если файл слишком большой, пережимаем
        if file_size > MAX_FILE_SIZE:
            logger.warning("⚠️ Файл слишком большой, пережимаю...")
            compressed_path = self.work_dir / "visualizer_compressed.mp4"
            (
                ffmpeg
                .input(str(output_path))
                .output(str(compressed_path), vcodec='libx264', preset='fast', crf=28, acodec='aac', bitrate='128k')
                .run(quiet=True, overwrite_output=True)
            )
            output_path = compressed_path
        
        return output_path
    
    def cleanup(self):
        """Удаление временных файлов"""
        try:
            shutil.rmtree(self.work_dir)
            logger.info(f"🗑️ Удалена папка: {self.work_dir}")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")

# ========== ТЕЛЕГРАМ-БОТ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветственное сообщение"""
    await update.message.reply_text(
        "🎵 **Music Visualizer Bot** 🎵\n\n"
        "Отправь мне **аудиофайл**, а я сделаю из него клип с визуализацией под бит!\n\n"
        "**Как это работает:**\n"
        "1️⃣ Анализирую музыку (нахожу биты, частоты)\n"
        "2️⃣ Создаю анимацию, реагирующую на ритм\n"
        "3️⃣ Отправляю готовый клип\n\n"
        "**Поддерживаемые форматы:** MP3, M4A, OGG, WAV\n"
        f"⏱ Максимальная длина: {MAX_DURATION} секунд",
        parse_mode=ParseMode.MARKDOWN
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка аудио"""
    audio = update.message.audio or update.message.voice
    if not audio:
        await update.message.reply_text("❌ Это не аудиофайл!")
        return
    
    status = await update.message.reply_text(
        "⏳ **Обрабатываю аудио...**\n"
        "• Анализирую биты\n"
        "• Создаю визуализацию\n"
        "• Генерирую видео\n\n"
        "Это может занять 20-60 секунд.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Сохраняем аудио
    ext = '.mp3' if update.message.audio else '.ogg'
    audio_path = TEMP_DIR / f"audio_{uuid.uuid4().hex}{ext}"
    
    file = await context.bot.get_file(audio.file_id)
    await file.download_to_drive(audio_path)
    
    visualizer = None
    try:
        # Создаём визуализатор
        visualizer = BeatVisualizer(audio_path)
        
        # Проверяем длительность
        if visualizer.duration > MAX_DURATION:
            await status.edit_text(
                f"⚠️ Аудио слишком длинное ({visualizer.duration:.0f} сек). "
                f"Я обрежу до {MAX_DURATION} секунд."
            )
            await asyncio.sleep(2)
        
        # Создаём видео
        await status.edit_text("🎨 **Генерирую видео...**", parse_mode=ParseMode.MARKDOWN)
        video_path = visualizer.create_video()
        
        # Отправляем результат
        await status.edit_text("📤 **Отправляю видео...**", parse_mode=ParseMode.MARKDOWN)
        
        with open(video_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=f"🎵 Клип под бит | BPM: {visualizer.tempo:.1f} | {visualizer.duration:.0f} сек",
                supports_streaming=True
            )
        
        await status.delete()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await status.edit_text(
            f"❌ **Ошибка:** {str(e)[:100]}\n"
            f"Попробуй другой файл или позже.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    finally:
        # Очистка
        if visualizer:
            visualizer.cleanup()
        if audio_path.exists():
            audio_path.unlink()

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена"""
    await update.message.reply_text("✅ Отменено. Можешь начать заново с /start")

# ========== ЗАПУСК ==========
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    
    print("\n" + "="*70)
    print("🎵 MUSIC VISUALIZER BOT 🎵")
    print("="*70)
    print(f"🤖 Бот: @{app.bot.username}")
    print(f"📁 Временная папка: {TEMP_DIR}")
    print(f"⏱ Макс длительность: {MAX_DURATION} сек")
    print("🎨 Режим: визуализация под бит")
    print("="*70)
    
    app.run_polling()

if __name__ == '__main__':
    main()
