"""Offline cue fusion: absence of evidence is never evidence of live play."""
import bisect
from .common import sample_frames

FIELDS = ['frame', 'time_s', 'state', 'evidence_score', 'whistle_recent',
    'flag_state', 'reason']


def timeline(audio, visual, start, end, fps, sample_hz=2.0):
    if not (0 <= start < end and fps > 0 and 0 < sample_hz <= fps):
        raise ValueError('Invalid timeline bounds/rate')
    whistles = sorted(audio, key=lambda r: float(r['start_s']))
    observations = sorted(visual, key=lambda r: (int(r['frame']), int(r['track'])))
    # Event closure needs a quiet gap and STFT context. We do not claim a
    # stoppage while an event is still being assembled by the offline detector.
    times = [float(r['start_s']) for r in whistles]
    frame_groups = {}
    for r in observations:
        frame_groups.setdefault(int(r['frame']), []).append(r)
    rows = []
    for frame in sample_frames(start, end, fps, sample_hz):
        t = frame/fps
        i = bisect.bisect_right(times, t)-1
        recent = None
        # No whistle evidence before the event has ended and closure context elapsed.
        while i >= 0 and t-float(whistles[i]['start_s']) <= 5.5:
            candidate = whistles[i]
            if 0.096 <= t-float(candidate['end_s']) <= 2.0:
                recent = candidate
                break
            i -= 1
        flags = frame_groups.get(frame, [])
        raised = [r for r in flags if r['state'] == 'RAISED_SUPPORTED']
        candidate_flags = [r for r in flags if r['state'] == 'RAISED_CANDIDATE']
        flag_state = ('RAISED_SUPPORTED' if raised else 'RAISED_CANDIDATE' if candidate_flags
            else 'LOWERED_SUPPORTED' if any(r['state'] == 'LOWERED_SUPPORTED' for r in flags)
            else 'LOWERED_CANDIDATE' if any(r['state'] == 'LOWERED_CANDIDATE' for r in flags)
            else 'UNKNOWN')
        state, score, reason = 'UNKNOWN', 0.0, 'no_current_stoppage_evidence_not_proof_of_live'
        if recent is not None:
            state, score, reason = 'STOPPAGE_CANDIDATE', float(recent['score']), 'whistle_may_also_mean_restart_or_spectator'
        if raised or candidate_flags:
            score = max(score, max(float(r['score']) for r in raised+candidate_flags))
            state, reason = 'STOPPAGE_CANDIDATE', 'flag_signal_does_not_itself_stop_play'
        if recent is not None and raised:
            state = 'STOPPAGE_SUPPORTED'
            score = min(float(recent['score']), max(float(r['score']) for r in raised))
            reason = 'coincident_whistle_and_persistent_flag_not_ground_truth'
        rows.append(dict(frame=frame, time_s=t, state=state, evidence_score=score,
            whistle_recent=int(recent is not None), flag_state=flag_state, reason=reason))
    return rows
