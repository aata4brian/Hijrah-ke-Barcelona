"""Flag/person association. Scores are evidence scores, never calibrated probabilities."""
import csv
from pathlib import Path

import cv2
import numpy as np

FIELDS = ['frame', 'time_s', 'track', 'state', 'score', 'reason',
    'person_x', 'person_y', 'person_w', 'person_h', 'flag_x', 'flag_y']


def color_candidates(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 115, 110), (38, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # Broad fixed thresholds, not tuned against target labels.
        if not (5 <= area <= 1200 and 2 <= w <= 65 and 2 <= h <= 65):
            continue
        if not (0.18 <= w/h <= 5 and area/(w*h) >= 0.22):
            continue
        found.append(dict(x=float(centers[i, 0]), y=float(centers[i, 1]),
            area=int(area), box=(int(x), int(y), int(w), int(h))))
    return sorted(found, key=lambda c: -c['area'])


def touchline_distance(matrix, point):
    if matrix is None or not np.isfinite(matrix).all():
        return None
    try:
        q = np.linalg.solve(matrix, np.r_[point, 1.0])
    except np.linalg.LinAlgError:
        return None
    if abs(q[2]) < 1e-9:
        return None
    x, y = q[:2]/q[2]
    if not (-0.08 <= x <= 1.08 and -0.12 <= y <= 1.12):
        return None
    return float(min(abs(y), abs(y-1)))


def near_touchline_proposals(matrix, candidates):
    """Vectorized proposal gate: one inverse per frame, not per colored blob."""
    if not candidates:
        return np.zeros(0, dtype=bool)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.zeros(len(candidates), dtype=bool)
    points = np.array([[[c['x'], c['y']+dy, 1] for dy in (0,30,60,100)]
        for c in candidates], dtype=float)
    projected = points @ inverse.T
    valid = np.isfinite(projected).all(axis=2) & (np.abs(projected[:,:,2]) > 1e-9)
    normalized = np.divide(projected[:,:,:2], projected[:,:,2,None],
        out=np.full(projected[:,:,:2].shape,np.nan), where=valid[:,:,None])
    x, y = normalized[:,:,0], normalized[:,:,1]
    valid &= (x >= -.08) & (x <= 1.08) & (y >= -.12) & (y <= 1.12)
    valid &= np.minimum(np.abs(y),np.abs(y-1)) <= .07
    return valid.any(axis=1)


def classify(person, candidates, matrix=None):
    x, y, w, h = person['box']
    k = np.asarray(person['keypoints'])
    feet = k[[15,16]]
    if np.all(feet[:, 2] >= 0.35):
        foot = feet[:, :2].mean(axis=0)
    else:
        foot = np.array([x+w/2, y+h])
    distance = touchline_distance(matrix, foot)
    if distance is None or distance > 0.045:
        return 'UNKNOWN', 0.0, 'touchline_identity_unavailable', None
    hypotheses = []
    for shoulder, elbow, wrist in [(5,7,9), (6,8,10)]:
        if np.min(k[[shoulder,elbow,wrist], 2]) < 0.4:
            continue
        s, e, hand = k[shoulder, :2], k[elbow, :2], k[wrist, :2]
        arm = np.linalg.norm(hand-e) + np.linalg.norm(e-s)
        if arm < 0.17*h:
            continue
        for flag in candidates:
            p = np.array([flag['x'], flag['y']])
            d = np.linalg.norm(p-hand)/h
            # Flag cloth is close to the hand/short flagstaff, not a shirt torso.
            if d > 0.48 or not 0.0003 <= flag['area']/(h*h) <= 0.11:
                continue
            if np.linalg.norm(p-s) < 0.14*h:
                continue
            # Boot/sock colors and shirt pixels are not cloth on a flagstaff.
            # Apply generic anatomical exclusions, never match-specific masks.
            visible_ankles = feet[feet[:, 2] >= 0.35, :2]
            if any(np.linalg.norm(p-ankle) < 0.14*h for ankle in visible_ankles):
                continue
            if x+0.22*w < p[0] < x+0.78*w and y+0.25*h < p[1] < y+0.65*h:
                continue
            raised = hand[1] < s[1]-0.10*h and p[1] < s[1]-0.06*h
            lowered = hand[1] > s[1]+0.25*h and p[1] > s[1]+0.25*h
            if not (raised or lowered):
                continue
            score = float(min(person['score'], float(k[[shoulder,elbow,wrist], 2].min()))
                * (1-0.5*d/0.48))
            item = ('RAISED_CANDIDATE' if raised else 'LOWERED_CANDIDATE', score,
                'pose_color_touchline_not_role_verified', flag)
            hypotheses.append(item)
    states = {item[0] for item in hypotheses}
    if len(states) > 1:
        return 'UNKNOWN', 0.0, 'contradictory_flag_geometries', None
    return max(hypotheses, key=lambda item:item[1]) if hypotheses else (
        'UNKNOWN', 0.0, 'no_unambiguous_hand_flag_geometry', None)


class HomographyReader:
    """Forward-only, explicitly frame-keyed homography with quality checks."""
    def __init__(self, path):
        self.handle = Path(path).open(encoding='utf-8-sig', newline='')
        self.reader = csv.DictReader(self.handle)
        self.current = None
        self.previous = -1
        self.requested = -1

    def at(self, frame):
        if frame < self.requested:
            raise ValueError('Homography lookup must be monotonic')
        self.requested = frame
        while self.current is None or int(self.current['frame']) < frame:
            self.current = next(self.reader, None)
            if self.current is None:
                return None
            index = int(self.current['frame'])
            if index <= self.previous:
                raise ValueError('Homography frame keys must be unique and sorted')
            self.previous = index
        if int(self.current['frame']) != frame:
            return None
        r = self.current
        if float(r['valid']) != 1 or not 0 <= float(r['reprojection_error']) <= 3:
            return None
        matrix = np.array([float(r[f'h{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        return matrix if np.isfinite(matrix).all() else None

    def close(self):
        self.handle.close()


class FlagDetector:
    def __init__(self, pose):
        self.pose = pose

    def __call__(self, image, matrix):
        candidates = color_candidates(image)
        if not candidates or matrix is None:
            return [], len(candidates), 0
        height, width = image.shape[:2]
        # Color proposals restrict expensive pose inference to small native-resolution tiles.
        proposals = []
        eligible = near_touchline_proposals(matrix, candidates)
        for c, allowed in zip(candidates, eligible):
            # A flag can be above the official. Project near its vertical lower neighborhood.
            if not allowed:
                continue
            if any(abs(c['x']-p[0]) < 150 and abs(c['y']-p[1]) < 120 for p in proposals):
                continue
            proposals.append((c['x'], c['y']))
            if len(proposals) >= 4:
                break
        result, seen = [], []
        for cx, cy in proposals:
            x = max(0, min(width-384, round(cx)-192))
            y = max(0, min(height-384, round(cy)-110))
            crop = image[y:min(height,y+384), x:min(width,x+384)]
            for p in self.pose(crop):
                p['box'][0] += x
                p['box'][1] += y
                p['keypoints'][:, :2] += [x,y]
                center = np.array([p['box'][0]+p['box'][2]/2, p['box'][1]+p['box'][3]/2])
                if any(np.linalg.norm(center-s) < p['box'][3]*0.4 for s in seen):
                    continue
                seen.append(center)
                state, score, reason, flag = classify(p, candidates, matrix)
                if state != 'UNKNOWN':
                    result.append(dict(person=p, state=state, score=score, reason=reason, flag=flag))
        # One colored component cannot independently identify several officials.
        counts = {}
        for detection in result:
            key = (detection['flag']['x'], detection['flag']['y'])
            counts[key] = counts.get(key, 0)+1
        result = [d for d in result if counts[(d['flag']['x'],d['flag']['y'])] == 1]
        return result, len(candidates), len(proposals)


class TemporalFlags:
    """Nearest-person association and time-aware persistence; no carry through occlusion."""
    def __init__(self, min_seconds=0.6, max_gap=1.6):
        self.min_seconds = min_seconds
        self.max_gap = max_gap
        self.tracks = {}
        self.next_id = 1
        self.last_time = -float('inf')

    def update(self, time, detections):
        if time <= self.last_time:
            raise ValueError('Flag observations must have strictly increasing times')
        self.last_time = time
        self.tracks = {i:r for i,r in self.tracks.items() if time-r['last'] <= self.max_gap}
        used, result = set(), []
        for d in detections:
            x,y,w,h = d['person']['box']
            foot = np.array([x+w/2,y+h])
            matches = [(np.linalg.norm(foot-r['foot'])/max(h,r['height']), i)
                for i,r in self.tracks.items() if i not in used]
            distance, ident = min(matches, default=(float('inf'), None))
            if distance > 0.8:
                ident = self.next_id
                self.next_id += 1
                self.tracks[ident] = dict(first=time, state=d['state'], count=0)
            r = self.tracks[ident]
            relative_flag = (np.array([d['flag']['x'], d['flag']['y']])-foot)/h
            flag_jump = ('relative_flag' in r and
                np.linalg.norm(relative_flag-r['relative_flag']) > 0.55)
            if r['state'] != d['state'] or flag_jump:
                r.update(first=time, state=d['state'], count=0)
            r.update(last=time, foot=foot, height=h, count=r['count']+1, relative_flag=relative_flag)
            stable = time-r['first'] >= self.min_seconds and r['count'] >= 2
            state = d['state'].replace('_CANDIDATE', '_SUPPORTED') if stable else d['state']
            used.add(ident)
            result.append(dict(d, track=ident, state=state))
        for ident in self.tracks.keys() - used:
            # Keep association briefly, but never accumulate persistence across missing observations.
            self.tracks[ident].update(count=0, first=time, state='UNKNOWN')
        return result
