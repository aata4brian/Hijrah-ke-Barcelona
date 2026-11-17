import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from deadball.common import sample_frames
from deadball.fusion import timeline
from deadball.media import probe, read_audio, audio_chunks
from deadball.video import observe

ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    def test_no_evidence_is_unknown_never_live(self):
        rows = timeline([], [], 0, 2, 25)
        self.assertEqual({r['state'] for r in rows}, {'UNKNOWN'})
        self.assertEqual([r['frame'] for r in rows], [0, 13, 25, 38])

    def test_only_completed_whistle_is_recent_and_expires(self):
        event = dict(start_s=.1, end_s=.8, score=.9)
        rows = timeline([event], [], 0, 4, 25)
        self.assertEqual(rows[1]['state'], 'UNKNOWN')
        self.assertEqual(rows[2]['state'], 'STOPPAGE_CANDIDATE')
        self.assertEqual(rows[-1]['state'], 'UNKNOWN')

    def test_both_cues_support_not_confirm_stoppage(self):
        flag = dict(frame=25, track=1, state='RAISED_SUPPORTED', score=.8)
        audio = [dict(start_s=.1, end_s=.8, score=.9)]
        rows = timeline(audio, [flag], 0, 2, 25)
        self.assertEqual(rows[2]['state'], 'STOPPAGE_SUPPORTED')
        self.assertEqual(rows[2]['evidence_score'], .8)
        self.assertEqual(rows[3]['flag_state'], 'UNKNOWN')

    def test_lowered_flag_not_proof_of_live(self):
        flag = dict(frame=0, track=1, state='LOWERED_SUPPORTED', score=.9)
        row = timeline([], [flag], 0, 1, 25)[0]
        self.assertEqual(row['state'], 'UNKNOWN')
        self.assertEqual(row['flag_state'], 'LOWERED_SUPPORTED')

    def test_sampling_noninteger_step_and_interval(self):
        frames = sample_frames(.01, 2.01, 25, 2)
        self.assertEqual(frames, [1, 14, 26, 39])
        self.assertEqual(len(set(frames)), len(frames))
        for bad in [(0, 0, 25, 2), (0, 1, 25, 0), (0, 1, 25, 26)]:
            with self.assertRaises(ValueError): sample_frames(*bad)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MediaIntegrationTests(unittest.TestCase):
    def test_visual_decode_alignment_and_unknown_without_detection(self):
        with tempfile.TemporaryDirectory(prefix='deadball_visual_') as directory:
            media = Path(directory)/'video.mp4'
            subprocess.run([shutil.which('ffmpeg'), '-nostdin', '-v', 'error',
                '-f', 'lavfi', '-i', 'color=c=green:s=320x240:r=25:d=2',
                '-c:v', 'mpeg4', str(media)], check=True, capture_output=True, timeout=20)
            def detector(image, matrix):
                self.assertEqual(image.shape,(240,320,3))
                self.assertIsNone(matrix)
                return [],0,0
            flags,audit = observe(media,probe(media),.1,1.9,detector,sample_hz=2)
            self.assertEqual(flags,[])
            self.assertEqual([r['frame'] for r in audit],sample_frames(.1,1.9,25,2))
            self.assertTrue(all(r['visual_state']=='UNKNOWN' for r in audit))
            self.assertTrue(all(abs(r['time_s']-r['frame']/25)<1e-9 for r in audit))

    def test_synthetic_audio_video_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='deadball_test_') as directory:
            root = Path(directory)
            media = root/'synthetic.mp4'
            command = [shutil.which('ffmpeg'), '-nostdin', '-v', 'error',
                '-f', 'lavfi', '-i', 'color=c=green:s=320x240:r=25:d=3',
                '-f', 'lavfi', '-i', r"aevalsrc=0.3*sin(2*PI*3200*t)*between(t\,0.6\,1.15):s=16000:d=3",
                '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(media)]
            subprocess.run(command, check=True, capture_output=True, timeout=25)
            info = probe(media)
            self.assertEqual(info['fps'], 25)
            self.assertGreater(info['audio_channels'], 0)
            samples = read_audio(media, 0, 2)
            self.assertEqual(samples.shape, (32000, 2))
            shifted = read_audio(media, .5, 2)
            self.assertEqual(shifted.shape, (32000, 2))
            # Check seeking does not silently shift AAC timing by one packet.
            np.testing.assert_allclose(shifted[:24000], samples[8000:], atol=.01)
            streamed = np.concatenate(list(audio_chunks(media,.5,2,chunk_s=1)))
            np.testing.assert_allclose(streamed,shifted,atol=1/32768)
            out = root/'result'
            call = [sys.executable, str(ROOT/'run.py'), 'audio', '--video', str(media),
                '--start', '0', '--duration', '3', '--out', str(out)]
            completed = subprocess.run(call, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['whistle_candidates'], 1)
            self.assertFalse(manifest['submission_changed'])
            with (out/'deadball_evidence.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertNotIn('LIVE', {r['state'] for r in rows})
            protected = (out/'manifest.json').read_bytes()
            repeated = subprocess.run(call, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((out/'manifest.json').read_bytes(), protected)

    def test_no_audio_is_explicit_not_silence(self):
        with tempfile.TemporaryDirectory(prefix='deadball_noaudio_') as directory:
            media = Path(directory)/'silent.mp4'
            subprocess.run([shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-f', 'lavfi',
                '-i', 'color=c=green:s=160x120:r=25:d=1', '-c:v', 'mpeg4', str(media)],
                check=True, capture_output=True, timeout=20)
            self.assertEqual(probe(media)['audio_channels'], 0)
            with self.assertRaises(RuntimeError): read_audio(media, 0, .5)

    def test_shorter_audio_stream_is_recorded_not_padded(self):
        with tempfile.TemporaryDirectory(prefix='deadball_shortaudio_') as directory:
            root = Path(directory)
            media = root/'shortaudio.mp4'
            subprocess.run([shutil.which('ffmpeg'), '-nostdin', '-v', 'error',
                '-f', 'lavfi', '-i', 'color=c=green:s=160x120:r=25:d=3',
                '-f', 'lavfi', '-i', 'sine=frequency=500:sample_rate=16000:duration=2',
                '-c:v', 'mpeg4', '-c:a', 'aac', str(media)],
                check=True,capture_output=True,timeout=25)
            info = probe(media)
            self.assertLess(info['audio_duration'],info['duration'])
            out = root/'result'
            result = subprocess.run([sys.executable,str(ROOT/'run.py'),'audio',
                '--video',str(media),'--out',str(out)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['audio_status'],'processed_partial_coverage')
            self.assertGreater(manifest['audio_audit']['uncovered_tail_seconds'],.8)
            self.assertLess(manifest['audio_audit']['decoded_samples'],48000)


if __name__ == '__main__': unittest.main()
