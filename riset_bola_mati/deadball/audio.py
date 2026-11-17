"""Spectral whistle candidates, not referee identity or ball state truth."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import stft

from .media import audio_chunks

FIELDS = ['start_s', 'end_s', 'score', 'peak_hz', 'channel', 'evidence']


@dataclass(frozen=True)
class WhistleConfig:
    sample_rate: int = 16000
    window: int = 512
    hop: int = 160
    low_hz: float = 1800
    high_hz: float = 5000
    min_snr_db: float = 15
    min_band_fraction: float = 0.18
    min_concentration: float = 0.32
    min_rms_db: float = -50
    min_duration: float = 0.09
    max_duration: float = 3.5
    gap_s: float = 0.07
    chunk_s: float = 20

    def __post_init__(self):
        if not (0 < self.hop <= self.window <= self.sample_rate):
            raise ValueError('Invalid STFT window/hop')
        if not (0 < self.low_hz < self.high_hz < self.sample_rate / 2):
            raise ValueError('Whistle band must be below Nyquist')
        if not (0 < self.min_duration <= self.max_duration and 1 <= self.chunk_s <= 60):
            raise ValueError('Invalid duration/chunk bounds')


def spectral_hits(samples, config=WhistleConfig()):
    """Return center time, channel, heuristic score, peak frequency per hop."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Audio must be finite [samples, channels]')
    if len(x) < config.window:
        return []
    hits = []
    for channel in range(x.shape[1]):
        freq, times, spectrum = stft(x[:, channel], fs=config.sample_rate,
            nperseg=config.window, noverlap=config.window-config.hop,
            boundary=None, padded=False)
        power = np.abs(spectrum) ** 2
        relevant = (freq >= 300) & (freq <= 7000)
        band = (freq >= config.low_hz) & (freq <= config.high_hz)
        bp = power[band]
        peak = bp.argmax(axis=0)
        top = bp[peak, np.arange(bp.shape[1])]
        noise = np.median(power[relevant], axis=0)
        snr = 10 * np.log10((top + 1e-16) / (noise + 1e-16))
        total = power[relevant].sum(axis=0) + 1e-16
        band_fraction = bp.sum(axis=0) / total
        # Sum around the peak, allowing finite-window leakage and mild modulation.
        concentrated = np.zeros_like(top)
        for delta in (-2, -1, 0, 1, 2):
            ix = peak + delta
            valid = (ix >= 0) & (ix < len(bp))
            concentrated[valid] += bp[ix[valid], np.flatnonzero(valid)]
        concentration = concentrated / (bp.sum(axis=0) + 1e-16)
        energy = np.r_[0.0, np.cumsum(x[:, channel].astype(np.float64)**2)]
        starts = np.arange(len(times)) * config.hop
        mean_square = (energy[starts+config.window]-energy[starts])/config.window
        rms_db = 10 * np.log10(mean_square + 1e-16)
        keep = ((snr >= config.min_snr_db) & (band_fraction >= config.min_band_fraction)
            & (concentration >= config.min_concentration) & (rms_db >= config.min_rms_db))
        score = (np.clip((snr - config.min_snr_db + 6) / 24, 0, 1)
            + np.clip(band_fraction, 0, 1) + np.clip(concentration, 0, 1)) / 3
        for i in np.flatnonzero(keep):
            hits.append((float(times[i]), channel, float(score[i]), float(freq[band][peak[i]])))
    return hits


def merge_hits(hits, start, end, config=WhistleConfig()):
    events = []
    for channel in sorted({h[1] for h in hits}):
        rows = sorted((h for h in hits if h[1] == channel), key=lambda r: r[0])
        groups = []
        for row in rows:
            if not groups or row[0] - groups[-1][-1][0] > config.gap_s + config.hop/config.sample_rate:
                groups.append([])
            groups[-1].append(row)
        for group in groups:
            a = max(start, group[0][0] - config.window/config.sample_rate/2)
            b = min(end, group[-1][0] + config.window/config.sample_rate/2)
            if not (config.min_duration <= b-a <= config.max_duration):
                continue
            # A sparse intermittent tone is not a continuous whistle.
            occupancy = len(group) * config.hop/config.sample_rate / max(b-a, 1e-9)
            if occupancy < 0.45:
                continue
            events.append(dict(start_s=a, end_s=b, score=float(np.mean([g[2] for g in group])),
                peak_hz=float(np.median([g[3] for g in group])), channel=str(channel),
                evidence='tonal_whistle_candidate_not_referee_verified'))
    # Deduplicate coincident stereo detections; do not average opposite-phase waveforms.
    result = []
    for event in sorted(events, key=lambda e: e['start_s']):
        duplicate = next((r for r in reversed(result) if min(r['end_s'], event['end_s'])
            - max(r['start_s'], event['start_s']) > 0.5 * min(r['end_s']-r['start_s'], event['end_s']-event['start_s'])), None)
        if duplicate is None:
            result.append(event)
        else:
            duplicate['start_s'] = min(duplicate['start_s'], event['start_s'])
            duplicate['end_s'] = max(duplicate['end_s'], event['end_s'])
            duplicate['score'] = max(duplicate['score'], event['score'])
            duplicate['channel'] = 'both'
    return result


def detect_samples(samples, config=WhistleConfig(), offset=0):
    duration = len(samples) / config.sample_rate
    hits = [(t+offset, c, s, f) for t,c,s,f in spectral_hits(samples, config)]
    return merge_hits(hits, offset, offset+duration, config)


def detect_media(path, start, duration, config=WhistleConfig(), progress=None):
    """Single audio decode; carry exact unconsumed STFT samples across chunks."""
    hits = []
    chunks = math.ceil(duration/config.chunk_s)
    decoded = 0
    cursor = 0
    tail = None
    for i, block in enumerate(audio_chunks(path,start,duration,config.sample_rate,config.chunk_s),1):
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        if tail is None:
            tail = np.empty((0,block.shape[1]), np.float32)
        decoded += len(block)
        audio = np.concatenate((tail,block))
        hits.extend((start+cursor/config.sample_rate+t,c,s,f)
            for t,c,s,f in spectral_hits(audio,config))
        windows = max(0,1+(len(audio)-config.window)//config.hop)
        used = windows*config.hop
        tail = audio[used:].copy()
        cursor += used
        if progress:
            progress(i, chunks)
    return merge_hits(hits, start, start+duration, config), dict(chunks=chunks, decoded_samples=decoded)
