"""Bounded FFmpeg reads, with video-zero-based media times."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import queue
import threading
import tempfile

import numpy as np


def binary(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} not on PATH; install FFmpeg or add its bin directory')
    return found


def probe(path, timeout=25):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run([binary('ffprobe'), '-v', 'error', '-show_entries',
        'format=duration,start_time:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,sample_rate,channels,start_time,duration',
        '-of', 'json', str(path)], capture_output=True, timeout=timeout, check=True)
    raw = json.loads(result.stdout)
    videos = [s for s in raw['streams'] if s['codec_type'] == 'video']
    audios = [s for s in raw['streams'] if s['codec_type'] == 'audio']
    if not videos:
        raise ValueError('Input has no video stream')
    video = videos[0]
    fps = float(Fraction(video['avg_frame_rate']))
    nominal = float(Fraction(video['r_frame_rate']))
    if fps <= 0 or abs(fps - nominal) > 0.001:
        raise ValueError('Variable/unknown frame rate: transcode to CFR before frame-level export')
    return dict(width=int(video['width']), height=int(video['height']), fps=fps,
        frames=int(video['nb_frames']) if str(video.get('nb_frames', '')).isdigit() else None,
        duration=float(raw['format']['duration']), video_start=float(video.get('start_time', 0)),
        audio_start=float(audios[0].get('start_time', 0)) if audios else None,
        audio_duration=float(audios[0]['duration']) if audios and audios[0].get('duration') else None,
        audio_channels=int(audios[0]['channels']) if audios else 0, streams=raw['streams'])


def read_audio(path, start, duration, sample_rate=16000, timeout=45):
    if start < 0 or duration <= 0 or duration > 62:
        raise ValueError('Audio reads must be positive and bounded to 62 seconds')
    # Output seeking is deliberate: input seeking shifted AAC timing in the
    # synthetic integration test. For long scans use audio_chunks (one process).
    command = [binary('ffmpeg'), '-nostdin', '-v', 'error', '-i', str(path),
        '-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-map', '0:a:0', '-vn', '-sn',
        '-ac', '2', '-ar', str(sample_rate), '-acodec', 'pcm_s16le', '-f', 's16le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    if len(result.stdout) % 4:
        raise RuntimeError('Incomplete PCM sample/channel pair')
    audio = np.frombuffer(result.stdout, dtype='<i2').reshape(-1, 2).astype(np.float32) / 32768
    expected = round(duration * sample_rate)
    if len(audio) < max(1, expected - round(0.03 * sample_rate)):
        raise RuntimeError('Decoded audio shorter than requested: do not invent silence/coverage')
    # Codec packets can extend past the requested endpoint. Never use samples
    # from a later chunk as if they belonged to the requested interval.
    return audio[:expected]


def audio_chunks(path, start, duration, sample_rate=16000, chunk_s=20, timeout=45):
    """One decode process, bounded queue; output seeking preserves AAC timestamps.

    Decode/discard pre-start audio once, not once per chunk. No full waveform
    file and no repeated remote seeks. Reader thread prevents unbounded blocking.
    """
    count = round(duration*sample_rate)
    if start < 0 or count <= 0 or not 1 <= chunk_s <= 60:
        raise ValueError('Invalid stream range/chunk size')
    command = [binary('ffmpeg'), '-nostdin', '-v', 'error', '-i', str(path),
        '-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-map', '0:a:0', '-vn', '-sn',
        '-ac', '2', '-ar', str(sample_rate), '-f', 's16le', 'pipe:1']
    messages = queue.Queue(maxsize=3)
    stopped = threading.Event()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        def put(item):
            while not stopped.is_set():
                try:
                    messages.put(item, timeout=.2)
                    return
                except queue.Full:
                    pass
        def reader():
            try:
                while block := process.stdout.read(round(chunk_s*sample_rate)*4):
                    put(block)
                put(None)
            except Exception as exc:
                put(exc)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        received = 0
        try:
            while True:
                try:
                    block = messages.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError('FFmpeg audio stream stalled') from None
                if block is None:
                    break
                if isinstance(block, Exception):
                    raise block
                if len(block) % 4:
                    raise RuntimeError('Incomplete stereo PCM packet')
                array = np.frombuffer(block, dtype='<i2').reshape(-1,2)
                array = array[:max(0,count-received)]
                received += len(array)
                if len(array):
                    yield array.astype(np.float32)/32768
            code = process.wait(timeout=timeout)
            if code:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace')[-1500:])
            if abs(received-count) > round(.03*sample_rate):
                raise RuntimeError(f'Audio coverage mismatch: expected {count}, decoded {received}')
        finally:
            stopped.set()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            thread.join(timeout=5)
            process.stdout.close()
