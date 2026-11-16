"""Pemulihan konservatif dari piksel video, dengan dua jangkar dan uji bolak-balik.

Tidak mengisi celah dengan garis lurus. Bola yang tertutup/tidak tampak boleh
tetap hilang. Hasil berupa sidecar; tracking asli tidak pernah ditulis ulang.
"""
import argparse
import csv
import json
import os
import tempfile
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .evaluate import atomic_json,write_csv


def project(matrix,point):
    z=matrix@np.r_[point,1.]
    return z[:2]/z[2] if abs(z[2])>1e-10 else np.full(2,np.nan)


def patch(frame,point,radius=6):
    h,w=frame.shape
    if not np.isfinite(point).all() or not (radius<=point[0]<w-radius and radius<=point[1]<h-radius):return None
    return cv2.getRectSubPix(frame,(radius*2+1,radius*2+1),tuple(map(float,point))).astype(np.float32)


def similarity(a,b):
    if a is None or b is None:return -1.
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a*b).sum()/den) if den>1e-5 else -1.


def follow(frames,start):
    points=[np.asarray(start,np.float32)];quality=[1.]
    template=patch(frames[0],start)
    if template is None or template.std()<4:return None,None,'weak_anchor'
    params=dict(winSize=(15,15),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
    for prev,cur in zip(frames[:-1],frames[1:]):
        p=points[-1].reshape(1,1,2)
        q,status,error=cv2.calcOpticalFlowPyrLK(prev,cur,p,None,**params)
        if q is None or not status[0,0]:return None,None,'flow_lost'
        back,ok,_=cv2.calcOpticalFlowPyrLK(cur,prev,q,None,**params)
        cycle=float(np.linalg.norm(back-p)) if back is not None else float('inf')
        ncc=similarity(template,patch(cur,q[0,0]))
        if ok is None or not ok[0,0] or cycle>1.5 or ncc<.35:return None,None,'appearance_or_cycle'
        points.append(q[0,0]);quality.append(float(np.exp(-cycle)*max(0,ncc)))
    return np.asarray(points),np.asarray(quality),''


def refine_anchor(frame,point,radius=12):
    """Find a compact bright component near a projected anchor, or abstain.

    Experimental image evidence, not a ball detector or a tracking truth label.
    No frame-specific rules; ambiguous candidates are deliberately rejected.
    """
    point=np.asarray(point,float)
    if patch(frame,point,radius+7) is None:return None,'outside_image'
    x,y=np.rint(point).astype(int);margin=radius+7
    roi=frame[y-margin:y+margin+1,x-margin:x+margin+1]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    bright=cv2.morphologyEx(roi,cv2.MORPH_TOPHAT,kernel)
    mask=(bright>=max(12.,float(np.percentile(bright,95)))).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for index in range(1,count):
        _,_,width,height,area=stats[index]
        if not (2<=area<=80 and 2<=width<=12 and 2<=height<=12
                and .45<=width/height<=2.2 and area/(width*height)>=.35):continue
        center=centers[index]+[x-margin,y-margin]
        distance=float(np.linalg.norm(center-point))
        if distance>radius:continue
        contrast=float(bright[labels==index].mean())
        candidates.append((distance,center,contrast))
    if not candidates:return None,'no_compact_candidate'
    candidates.sort(key=lambda item:item[0])
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<4.:
        return None,'ambiguous_candidates'
    return candidates[0][1].astype(np.float32),''


def recover_gap(frames,matrices,valid,reprojection,start,end,pitch,refine=False):
    n=len(frames)
    if n<3 or len(matrices)!=n:return None,None,'length'
    if not np.all(valid) or not np.isfinite(matrices).all() or not np.isfinite(reprojection).all() or np.max(reprojection)>3.:
        return None,None,'homography'
    a=project(matrices[0],start/pitch);b=project(matrices[-1],end/pitch)
    if refine:
        for side,index,anchor in [('forward',0,a),('reverse',n-1,b)]:
            corrected,why=refine_anchor(frames[index],anchor)
            if corrected is None:return None,None,side+'_anchor_'+why
            try:offset=np.linalg.norm(project(np.linalg.inv(matrices[index]),corrected)*pitch-([start,end][index!=0]))
            except np.linalg.LinAlgError:return None,None,'singular_homography'
            if offset>.75:return None,None,side+'_anchor_metric_offset'
            if index==0:a=corrected
            else:b=corrected
    forward,fq,why=follow(frames,a)
    if forward is None:return None,None,'forward_'+why
    reverse,rq,why=follow(frames[::-1],b)
    if reverse is None:return None,None,'reverse_'+why
    backward=reverse[::-1];rq=rq[::-1]
    discrepancy=np.linalg.norm(forward-backward,axis=1)
    if discrepancy.max()>4.:return None,None,'two_anchor_disagreement'
    # Choose independently tracked pixel positions; this is not pitch interpolation.
    pixels=np.where((fq>=rq)[:,None],forward,backward)
    try:xy=np.asarray([project(np.linalg.inv(h),p)*pitch for h,p in zip(matrices,pixels)])
    except np.linalg.LinAlgError:return None,None,'singular_homography'
    xy[0]=start;xy[-1]=end
    if not np.isfinite(xy).all() or (xy < -3).any() or (xy>pitch+3).any():return None,None,'outside_pitch'
    if np.linalg.norm(np.diff(xy,axis=0),axis=1).max()*25>45.:return None,None,'implausible_speed'
    confidence=np.exp(-(discrepancy/3.)**2)*np.minimum(fq,rq)
    if np.min(confidence[1:-1])<.2:return None,None,'low_confidence'
    return xy,confidence,'accepted'


def gaps(ball,first,last):
    missing=~np.isfinite(ball[first:last]).all(axis=1)
    edges=np.diff(np.r_[False,missing,False].astype(np.int8))
    for a,b in zip(np.flatnonzero(edges==1)+first,np.flatnonzero(edges==-1)+first):
        yield int(a),int(b)


class RejectedGap(Exception):pass


def read_frames(cap,first,last,anchor=None,buffer_path=None,refine=False):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES,first):raise RuntimeError('Video seek failed')
    frames=[];disk=None
    try:
        for index in range(first,last):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Video decode failed at {index}')
            actual=cap.get(cv2.CAP_PROP_POS_FRAMES)
            if actual and abs(actual-(index+1))>.5:raise RuntimeError('Video frame alignment failed')
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if buffer_path is None:
                frames.append(gray)
            else:
                if disk is None:
                    disk=np.lib.format.open_memmap(buffer_path,mode='w+',dtype=np.uint8,shape=(last-first,*gray.shape))
                disk[index-first]=gray
                frames=disk[:index-first+1]
            # Fail early on bad pixels instead of buffering ten seconds of unusable
            # video on a laptop. The complete anchored test still runs afterwards.
            if anchor is not None:
                if len(frames)==1:
                    if refine:
                        anchor,reason=refine_anchor(gray,anchor)
                        if anchor is None:raise RejectedGap('forward_anchor_'+reason)
                    template=patch(gray,anchor)
                    if template is None or template.std()<4:raise RejectedGap('forward_weak_anchor')
                    point=anchor
                else:
                    tracked,_,reason=follow(frames[-2:],point)
                    if tracked is None:raise RejectedGap('forward_'+reason)
                    point=tracked[-1]
                if buffer_path is None and len(frames)*gray.nbytes>96*1024*1024:raise RejectedGap('video_memory_budget')
    except Exception:
        if disk is not None:disk._mmap.close()
        raise
    return frames


def recover_window(cap,a,b,matrices,valid,error,ball,pitch,refine=False):
    if not np.all(valid[a-1:b+1]) or np.max(error[a-1:b+1])>3:
        return None,None,'homography'
    frames=None
    try:
        with tempfile.TemporaryDirectory(prefix='mtah_video_') as temp:
            try:
                anchor=project(matrices[a-1],ball[a-1]/pitch)
                frames=read_frames(cap,a-1,b+1,anchor,Path(temp)/'gray.npy',refine=refine)
                return recover_gap(frames,matrices[a-1:b+1],valid[a-1:b+1],error[a-1:b+1],ball[a-1],ball[b],pitch,refine=refine)
            finally:
                if isinstance(frames,np.memmap):frames._mmap.close()
    except RejectedGap as rejection:return None,None,str(rejection)
    except (RuntimeError,cv2.error,OSError) as failure:return None,None,'video_io_error: '+str(failure)


def run(args):
    refine=getattr(args,'refine_anchors',False)
    # Set before importing games: that module resolves DATA_DIR on import.
    import games
    games.DATA_DIR=str(args.data_dir)
    from games import GAMES,GAME,half_video_blocks
    from night_tracking import local_halves
    from starter.load_homography import load
    from .data import fingerprint
    args.output.mkdir(parents=True,exist_ok=True)
    n=GAMES[GAME]['n_video_frames'];pitch=np.asarray(GAMES[GAME]['pitch'])
    ball=np.full((n,2),np.nan,np.float32)
    for first,last,data,_ in local_halves():ball[first:last]=data.ball
    original=ball.copy();confidence=np.zeros(n,np.float32);accepted=np.zeros(n,bool)
    matrices,error,valid=load(data_dir=str(args.data_dir))
    from games import asset_paths
    hp=Path(asset_paths(data_dir=str(args.data_dir))['homography'])
    import pandas as pd
    indices=pd.read_csv(hp,usecols=['frame'])['frame'].to_numpy()
    if len(matrices)!=n or not np.array_equal(indices,np.arange(n)):raise ValueError('Homography frame alignment mismatch')
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(str(args.video),cv2.CAP_FFMPEG,
                         [cv2.CAP_PROP_N_THREADS,1,cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,20000,
                          cv2.CAP_PROP_READ_TIMEOUT_MSEC,20000])
    if not cap.isOpened():raise FileNotFoundError(f'Cannot open video {args.video}')
    if abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=n:
        cap.release();raise ValueError('Expected tactical video: 25 fps, 199319 frames')
    candidates=[]
    entropy=None
    if args.probabilities:
        p=np.load(args.probabilities,allow_pickle=False)['probabilities']
        if p.shape!=(n,3):raise ValueError('Probability frame alignment mismatch')
        entropy=-(p*np.log(np.clip(p,1e-7,1))).sum(axis=1)
    for _,first,last in half_video_blocks(GAME):
        for a,b in gaps(ball,first,last+1):
            duration=b-a
            priority=duration*(1+float(entropy[a:b].mean()) if entropy is not None else 1.)
            reason='pending'
            if a<=first or b>last:reason='missing_anchor'
            elif duration>args.max_gap_frames:reason='too_long_for_reliable_flow'
            elif a<args.range_start or b>args.range_end:reason='outside_requested_range'
            candidates.append(dict(start=a,end=b,length=duration,priority=priority,status=reason,recovered_frames=0))
    candidates.sort(key=lambda v:-v['priority'])
    identity=dict(video=str(args.video.resolve()),video_bytes=args.video.stat().st_size,
                  video_mtime_ns=args.video.stat().st_mtime_ns,tracking_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                  homography_sha256=fingerprint(hp),code_sha256=fingerprint(Path(__file__)),top=args.top,
                  max_gap_frames=args.max_gap_frames,selftest=args.selftest,range_start=args.range_start,range_end=args.range_end,
                  probabilities_sha256=fingerprint(args.probabilities) if args.probabilities else None)
    if refine:identity['refine_anchors']=True
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity:
        cap.release();raise ValueError('Video output belongs to another configuration; use a new output directory')
    atomic_json(manifest,identity)
    processed=0;selftest=[]
    checkpoint=args.output/'recovery_checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as old:
            ball[:]=old['ball'];confidence[:]=old['confidence'];accepted[:]=old['accepted']
            saved=json.loads(str(old['progress']))
        selftest=[r for r in saved['selftest'] if not r['status'].startswith('video_io_error')]
        prior={r['start']:r for r in saved['candidates']}
        for row in candidates:
            if row['start'] in prior:row.update(prior[row['start']])
        processed=saved['processed']
        for row in candidates:
            if row['status'].startswith('video_io_error'):
                row['status']='pending';processed-=1
    def save_progress():
        temp=checkpoint.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,ball=ball,confidence=confidence,accepted=accepted,
                progress=json.dumps(dict(selftest=selftest,candidates=candidates,processed=processed)))
        temp.replace(checkpoint)
    done_selftest={r['start'] for r in selftest}
    try:
        # Mask observed coordinates only for this diagnostic; never use them as possession labels.
        for _,first,last in half_video_blocks(GAME):
            for a in range(first+50,last-20,487):
                b=a+5
                if a<args.range_start or b>args.range_end:continue
                if len(selftest)>=args.selftest:break
                if a in done_selftest:continue
                if not np.isfinite(original[a-1:b+1]).all():continue
                xy,c,why=recover_window(cap,a,b,matrices,valid,error,original,pitch,refine=refine)
                row=dict(start=a,end=b,status=why,mean_error_m='',max_error_m='')
                if xy is not None:
                    distance=np.linalg.norm(xy[1:-1]-original[a:b],axis=1)
                    row.update(mean_error_m=float(distance.mean()),max_error_m=float(distance.max()))
                selftest.append(row);save_progress()
                write_csv(args.output/'selftest_tracking_agreement.csv',selftest)
                print(json.dumps(dict(stage='video_selftest',**row)),flush=True)
        for row in candidates:
            if row['status']!='pending':continue
            if processed>=args.top:row['status']='not_attempted_budget';continue
            processed+=1;a,b=row['start'],row['end']
            xy,c,why=recover_window(cap,a,b,matrices,valid,error,ball,pitch,refine=refine)
            row['status']=why
            if xy is not None:
                ball[a:b]=xy[1:-1];confidence[a:b]=c[1:-1];accepted[a:b]=True
                row['recovered_frames']=b-a
            save_progress()
            print(json.dumps(dict(stage='video_gap',**row)),flush=True)
            write_csv(args.output/'gap_audit.csv',candidates)
            atomic_json(args.output/'status.json',dict(stage='recovering',processed=processed,recovered_frames=int(accepted.sum())))
    finally:cap.release()
    if not np.array_equal(ball[np.isfinite(original).all(1)],original[np.isfinite(original).all(1)]):
        raise AssertionError('Observed tracking was changed')
    np.savez_compressed(args.output/'ball_recovery.npz',ball=ball,confidence=confidence,recovered=accepted,fps=25,
                        original_ball=original)
    if candidates:write_csv(args.output/'gap_audit.csv',candidates)
    if selftest:write_csv(args.output/'selftest_tracking_agreement.csv',selftest)
    io_errors=sum(r['status'].startswith('video_io_error') for r in candidates+selftest)
    result=dict(complete=io_errors==0,video_io_errors=io_errors,attempted_gaps=processed,total_gaps=len(candidates),recovered_frames=int(accepted.sum()),
                accepted_gaps=sum(r['status']=='accepted' for r in candidates),
                selftest_attempts=len(selftest),selftest_accepted=sum(r['status']=='accepted' for r in selftest),
                homography_sha256=fingerprint(hp),video_name=args.video.name,video_bytes=args.video.stat().st_size,
                max_gap_frames=args.max_gap_frames,top=args.top,refine_anchors=refine,
                requested_range=[args.range_start,args.range_end],code_sha256=fingerprint(Path(__file__)),
                limitation='Agreement with supplied tracking is a diagnostic, not human ground truth. Recovery is not automatically promoted to submission.')
    atomic_json(args.output/'result.json',result)
    atomic_json(args.output/'status.json',dict(stage='complete' if result['complete'] else 'incomplete',**result))
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--top',type=int,default=40)
    parser.add_argument('--max-gap-frames',type=int,default=250);parser.add_argument('--selftest',type=int,default=20)
    parser.add_argument('--probabilities',type=Path)
    parser.add_argument('--refine-anchors',action='store_true',
                        help='Experimental local compact-component anchor refinement; use a new output folder')
    parser.add_argument('--range-start',type=int,default=0)
    parser.add_argument('--range-end',type=int,default=199319)
    args=parser.parse_args()
    if min(args.top,args.max_gap_frames,args.selftest)<0:parser.error('Budgets must be nonnegative')
    if not 0<=args.range_start<args.range_end<=199319:parser.error('Invalid tactical frame range')
    run(args)


if __name__=='__main__':main()
