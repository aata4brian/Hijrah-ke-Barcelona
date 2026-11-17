import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from deadball.audio import WhistleConfig, detect_samples, detect_media, spectral_hits


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.config = WhistleConfig()
        self.sr = self.config.sample_rate

    def signal(self, frequency=3200, start=0.6, stop=1.15, duration=2):
        t = np.arange(round(duration*self.sr))/self.sr
        rng = np.random.default_rng(17)
        x = rng.normal(0,0.002,len(t))
        active = (t >= start) & (t < stop)
        x[active] += 0.25*np.sin(2*np.pi*frequency*t[active])
        return x.astype(np.float32)

    def test_detect_tonal_burst_and_timing(self):
        events = detect_samples(self.signal())
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0]['start_s'],0.6,delta=0.04)
        self.assertAlmostEqual(events[0]['end_s'],1.15,delta=0.04)
        self.assertAlmostEqual(events[0]['peak_hz'],3200,delta=40)

    def test_antiphase_stereo_not_cancelled(self):
        x = self.signal()
        events = detect_samples(np.column_stack((x,-x)))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['channel'],'both')

    def test_silence_and_broadband_noise_rejected(self):
        self.assertEqual(detect_samples(np.zeros(self.sr)),[])
        noise = np.random.default_rng(5).normal(0,0.1,self.sr*2)
        self.assertEqual(detect_samples(noise),[])

    def test_low_pitch_and_short_click_rejected(self):
        self.assertEqual(detect_samples(self.signal(frequency=500)),[])
        self.assertEqual(detect_samples(self.signal(start=.6,stop=.61)),[])

    def test_continuous_tone_over_maximum_is_rejected(self):
        x = self.signal(start=0,stop=5,duration=5)
        self.assertEqual(detect_samples(x),[])

    def test_offset_is_preserved(self):
        event = detect_samples(self.signal(),offset=100)[0]
        self.assertAlmostEqual(event['start_s'],100.6,delta=.04)

    def test_invalid_samples_rejected(self):
        for x in [np.array([np.nan]),np.zeros((2,2,2))]:
            with self.assertRaises(ValueError):
                spectral_hits(x)

    def test_chunk_boundary_event_not_duplicated(self):
        config = replace(self.config,chunk_s=1)
        signal = self.signal(start=.85,stop=1.25,duration=3)
        def chunks(*args):
            for a in range(0,len(signal),self.sr):
                yield signal[a:a+self.sr,None]
        with patch('deadball.audio.audio_chunks',side_effect=chunks):
            events,audit = detect_media('unused',0,3,config)
        self.assertEqual(audit['chunks'],3)
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0]['start_s'],.85,delta=.05)
        self.assertAlmostEqual(events[0]['end_s'],1.25,delta=.05)

    def test_config_checks(self):
        for options in [dict(low_hz=9000),dict(hop=1000),dict(chunk_s=61)]:
            with self.assertRaises(ValueError):
                WhistleConfig(**options)


if __name__ == '__main__':
    unittest.main()
