import sys, os
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
SRC=Path(r"E:/data/images/unknown_pos150"); DST=Path(r"E:/data/images/unknown_pos150_384"); SZ=384
def one(args):
    src,dst=args
    try:
        if dst.exists(): return 1
        im=Image.open(src).convert("RGB").resize((SZ,SZ),Image.BILINEAR)
        im.save(dst); return 1
    except Exception as e: return 0
def main():
    jobs=[]
    for cd in sorted(SRC.iterdir()):
        if cd.is_dir():
            out=DST/cd.name; out.mkdir(parents=True,exist_ok=True)
            for p in cd.glob("*.png"): jobs.append((p, out/p.name))
    print(f"[preresize] {len(jobs)} imgs, {len(list(DST.iterdir()))} classes -> {DST}",flush=True)
    with Pool(8) as pool:
        done=0
        for r in pool.imap_unordered(one, jobs, chunksize=16):
            done+=1
            if done%500==0: print(f"  {done}/{len(jobs)}",flush=True)
    print(f"[done] {done}",flush=True)
if __name__=="__main__": main()
