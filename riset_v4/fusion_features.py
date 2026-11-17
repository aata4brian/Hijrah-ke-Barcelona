"""Label-free V4.6 features; leave the frozen 203-column schema unchanged."""
import numpy as np
from night_model import mean_valid, rolling
from .features import FPS, velocity

SCHEMA = 'v46-collective-restart-quality-v1'


def shifted(values, frames):
    """NaN padding, never wrap one half's endpoints into the other."""
    result = np.full_like(values, np.nan, dtype=np.float32)
    if frames == 0:
        return np.asarray(values, dtype=np.float32).copy()
    if abs(frames) < len(values):
        if frames > 0:
            result[frames:] = values[:-frames]
        else:
            result[:frames] = values[-frames:]
    return result


def positive_run(mask):
    indices = np.arange(len(mask))
    breaks = np.maximum.accumulate(np.where(~mask, indices, -1))
    return np.minimum(indices-breaks, 250)/FPS


def extra_pair(data):
    """Offline context, metre coordinates; player slots are NOT persistent IDs."""
    ball = np.asarray(data.ball, dtype=np.float32)
    n = len(ball)
    if ball.shape != (n, 2) or n == 0:
        raise ValueError('Expected nonempty ball [frames,2]')
    if len(data.pitch) != 2 or not np.isfinite(data.pitch).all() or np.any(data.pitch <= 0):
        raise ValueError('Invalid pitch')
    seen = np.isfinite(ball).all(1)
    speed = np.linalg.norm(velocity(ball), axis=1)
    edge = np.minimum(ball, data.pitch-ball)
    shared = [np.min(edge, axis=1), np.minimum(edge[:, 0], 15),
              np.minimum(edge[:, 1], 15), positive_run(seen & (speed < .8))]
    names = ['line_distance_signed', 'endline_distance', 'touchline_distance', 'ball_static_run']
    shared[3][~np.isfinite(speed)] = np.nan
    team_cols = []
    for players in data.players:
        players = np.asarray(players, dtype=np.float32)
        valid = np.isfinite(players).all(2)
        pv = velocity(players, lag=5, limit=12)
        ps = np.linalg.norm(pv, axis=2)
        observed_motion = np.isfinite(ps)
        static = np.where(observed_motion, ps < 1, np.nan)
        fast = np.where(observed_motion, ps > 3, np.nan)
        distances = np.linalg.norm(players-ball[:, None], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        nearest = distances.min(1)
        nearest[~np.isfinite(nearest)] = np.nan
        density = (distances < 2.5).sum(1)/np.maximum(valid.sum(1), 1)
        density[~seen | (valid.sum(1) == 0)] = np.nan
        cols = [valid.mean(1), observed_motion.mean(1), mean_valid(ps),
                mean_valid(static), mean_valid(fast), nearest,
                density]
        tn = ['visible_share', 'motion_coverage', 'speed', 'static_share',
              'fast_share', 'nearest', 'contact_density']
        # Unit-direction coherence is identity-free, but motion still rejects slot jumps.
        unit = pv/np.maximum(ps[:, :, None], .2)
        unit[~observed_motion] = np.nan
        coherence = np.linalg.norm(np.stack([mean_valid(unit[:, :, k]) for k in (0, 1)], 1), axis=1)
        cols.append(coherence); tn.append('motion_coherence')
        for w in (25, 75):
            for key, values in [('speed', cols[2]), ('static', cols[3]), ('contact', cols[6])]:
                cols.append(rolling(values, w)); tn.append(f'{key}_mean_{w}')
        for key, values in [('speed', cols[2]), ('static', cols[3]), ('contact', cols[6])]:
            cols += [shifted(values, 25), shifted(values, -25),
                     shifted(values, -25)-shifted(values, 25)]
            tn += [f'{key}_before1s', f'{key}_after1s', f'{key}_change2s']
        team_cols.append(np.column_stack(cols))
    home, away = team_cols
    # Team-invariant interactions. Missing tracking remains missing, not zero speed.
    both = np.isfinite(home[:, 5]) & np.isfinite(away[:, 5])
    contest = np.where(both, np.exp(-np.maximum(home[:, 5], away[:, 5])/3), np.nan)
    advantage = np.where(both, np.abs(home[:, 5]-away[:, 5]), np.nan)
    collective = (home[:, 3]+away[:, 3])/2
    shared += [contest, advantage, collective,
               shifted(collective, -25)-shifted(collective, 25),
               rolling(collective, 75)*np.exp(-np.maximum(speed, 0)/2)]
    names += ['contested_contact', 'contact_advantage_abs', 'collective_static',
              'collective_stop_start_change', 'collective_stop_ball_interaction']
    s = np.column_stack(shared)
    arrays = (np.column_stack([s, home, away, home-away]),
              np.column_stack([s, away, home, away-home]))
    arrays = tuple(np.where(np.isfinite(x), x, np.nan).astype(np.float32) for x in arrays)
    names += [prefix+name for prefix in ('home_', 'away_', 'diff_') for name in tn]
    return arrays, names
