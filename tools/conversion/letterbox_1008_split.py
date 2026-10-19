"""生成独立224方形版本，只对非224视频做letterbox，不改划分与数值。

默认参数保持 1008 历史行为；其他批次用 --source/--out 指定划分根目录与新版本目录。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time
import av

BASE=Path('output/1008_pick_apple_20261008')
SOURCE=BASE/'training_split_v1'
OUT=BASE/'training_split_letterbox224_v1'

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def encode(job):
    src,dst,expected,split,episode,camera=job
    with av.open(str(src)) as c:
        stream=c.streams.video[0];w,h=stream.codec_context.width,stream.codec_context.height
    scale=224/max(w,h);nw,nh=round(w*scale),round(h*scale);x,y=(224-nw)//2,(224-nh)//2
    vf=f'scale={nw}:{nh}:flags=bilinear,pad=224:224:{x}:{y}:color=black,setsar=1'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(src),'-vf',vf,'-fps_mode','passthrough','-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-an',str(dst)],check=True,capture_output=True)
    with av.open(str(dst)) as c:
        stream=c.streams.video[0]
        assert (stream.codec_context.width,stream.codec_context.height)==(224,224)
        count=sum(1 for _ in c.decode(stream))
    assert count==expected,(src,count,expected)
    print('LETTERBOX',split,episode,camera,count,flush=True)
    return dict(split=split,episode=episode,camera=camera,source_shape=[h,w,3],target_shape=[224,224,3],resized_shape=[nh,nw,3],padding_left=x,padding_top=y,padding_right=224-nw-x,padding_bottom=224-nh-y,interpolation='bilinear',frames=count)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE, help='train/val 划分根目录')
    parser.add_argument('--out', type=Path, default=OUT, help='letterbox 新版本输出目录（不得已存在）')
    args = parser.parse_args()
    source_root, out_root = args.source, args.out
    started=time.monotonic()
    out_root.mkdir(parents=True,exist_ok=False)
    dump(out_root/'status.json',dict(phase='converting',started_unix=time.time()))
    try:
        jobs=[];proof=[];unchanged_videos=0
        for split in ['train','val']:
            source=source_root/split;dest=out_root/split
            shutil.copytree(source/'meta',dest/'meta')
            # 新视频统计尚未计算，禁止沿用旧验收状态。
            info=json.loads((dest/'meta/info.json').read_text(encoding='utf-8'))
            info['statistics_status']='pending recomputation after letterbox'
            info.pop('normalization_statistics_sha256',None)
            for key,v in info['features'].items():
                if v['dtype']=='video':
                    v['shape']=[224,224,3]
                    v['names']=['height','width','channels']
                    v['info']={'video.height':224,'video.width':224,'video.channels':3,'video.fps':30.,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}
            info['letterbox_provenance']=dict(source=str(source_root),reencode_rule='非224视频等比例缩放加黑边；已为224的视频原字节复制',target=[224,224],reason='用户指定与已有模型图像预处理一致')
            dump(dest/'meta/info.json',info)
            eps=[json.loads(l) for l in (source/'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines() if l]
            for ep in eps:
                i,n=ep['episode_index'],ep['length']
                rel=Path(f'data/chunk-000/episode_{i:06d}.parquet')
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,target)
                before=hashlib.sha256((source/rel).read_bytes()).hexdigest()
                assert hashlib.sha256(target.read_bytes()).hexdigest()==before
                proof.append(dict(split=split,episode=i,sha256=before,original_episode_id=ep['original_episode_id']))
                for key,v in info['features'].items():
                    if v['dtype']!='video':continue
                    rel=Path(f'videos/chunk-000/{key}/episode_{i:06d}.mp4')
                    src_video=source/rel;target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                    with av.open(str(src_video)) as probe:
                        st=probe.streams.video[0]
                        already_224=(st.codec_context.width,st.codec_context.height)==(224,224)
                    if already_224:
                        shutil.copy2(src_video,target);unchanged_videos+=1
                    else:
                        jobs.append((src_video,target,n,split,i,key))
        with ProcessPoolExecutor(max_workers=4) as pool:geometry=list(pool.map(encode,jobs))
        manifest=json.loads((source_root/'split_manifest.json').read_text(encoding='utf-8'))
        manifest['statistics_status']='pending';manifest['letterbox_applied']=True
        dump(out_root/'split_manifest.json',manifest)
        dump(out_root/'letterbox_geometry.json',geometry)
        dump(out_root/'numeric_identity.json',dict(all_parquet_byte_identical=True,episodes=proof))
        dump(out_root/'status.json',dict(phase='converted_pending_validation',new_videos=len(jobs),unchanged_videos=unchanged_videos,elapsed_seconds=time.monotonic()-started))
        print('LETTERBOX_DONE',len(jobs),'unchanged',unchanged_videos,flush=True)
    except Exception as exc:
        dump(out_root/'status.json',dict(phase='failed',error=str(exc)))
        raise

if __name__=='__main__':main()
