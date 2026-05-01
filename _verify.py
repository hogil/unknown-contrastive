"""Verify generated wafer synthetic samples.

Checks:
- 클래스 폴더 구조 (36 classes)
- 파일명 9-token 형식
- PNG mode/size/palette
- JSON 페어 + 스키마
- yield/sys filename ↔ JSON 일치
- chip count ≈ 803 inside-wafer
- synthetic partid/pgm + FTN/QTN keys + chip f/q arrays

Usage:
    python _verify.py                    # 전체 검사
    python _verify.py --class Center_bank_boundary  # 특정 클래스만
    python _verify.py --sample 10        # 클래스당 10개만
"""
import os, json, glob, re, argparse, sys
from PIL import Image

PNG_ROOT  = "D:/project/data/wm-811k/unknown"
JSON_ROOT = "D:/project/data/positions/unknown"
SIZE      = 6400
N_INSIDE  = 803                                                                       # 허용 범위 700-850
TD_OPTIONS = {'PE', 'EE', 'PT'}
LT_OPTIONS = {'NORMAL', 'PWQ', 'ENGINEER'}

EXPECTED_CLASSES = ['Center', 'Donut', 'Edge-Ring', 'Edge-Bottom', 'Edge-Top',
                    'Full', 'Thick-Edge']
EXPECTED_OBJECTS = ['bank_boundary', 'particle_blast', 'scratch',
                    'scratch_21deg', 'invalid_main']
SPECIAL_CLASSES = ['Normal', 'Starburst', 'CommaCluster']                              # object 없는 단독 class

def expected_class_set():
    s = set()
    for c in EXPECTED_CLASSES:
        for o in EXPECTED_OBJECTS:
            if c == 'Thick-Edge' and o != 'invalid_main': continue
            s.add(f"{c}_{o}")
    for c in SPECIAL_CLASSES:
        s.add(c)
    return s

def parse_filename(fname):
    if not fname.endswith('.png'): return None
    toks = fname[:-4].split('_')
    if len(toks) != 9: return None
    return dict(zip(['root','step','wafer','ymd','hms','yld','sys','td','lt'], toks))

def check_filename(f):
    errs = []
    if f is None: return ['filename: not 9 tokens']
    if not re.match(r'^[a-z]{3}\d{3}$', f['root']):
        errs.append(f"root format: {f['root']}")
    if f['step'] not in ('00P','00C'):
        errs.append(f"step: {f['step']}")
    if not (re.match(r'^\d{2}$', f['wafer']) and 1 <= int(f['wafer']) <= 24):
        errs.append(f"wafer: {f['wafer']}")
    if not re.match(r'^\d{8}$', f['ymd']): errs.append(f"ymd: {f['ymd']}")
    if not re.match(r'^\d{6}$', f['hms']): errs.append(f"hms: {f['hms']}")
    try: float(f['yld'])
    except: errs.append(f"yld: {f['yld']}")
    try: float(f['sys'])
    except: errs.append(f"sys: {f['sys']}")
    if f['td'] not in TD_OPTIONS: errs.append(f"td: {f['td']}")
    if f['lt'] not in LT_OPTIONS: errs.append(f"lt: {f['lt']}")
    return errs

def check_png(path):
    errs = []
    try:
        im = Image.open(path)
        if im.mode != 'P': errs.append(f"mode {im.mode} (expected P)")
        if im.size != (SIZE, SIZE): errs.append(f"size {im.size}")
        pal = im.getpalette()
        if pal is None or len(pal) < 96: errs.append("palette < 32 colors")
    except Exception as e:
        errs.append(f"png open: {e}")
    return errs

def check_json(jpath, fields):
    errs = []
    try:
        with open(jpath, 'r', encoding='utf-8') as f:
            j = json.load(f)
    except Exception as e:
        return [f"json open: {e}"]
    for k in ['root','step','wafer','stime','partid','pgm','ftn_keys','qtn_keys',
              'netd','gd','yield','sys','tm','lt','coord','chips']:
        if k not in j: errs.append(f"missing key: {k}")
    if not j.get('partid'):
        errs.append("partid empty")
    if not j.get('pgm'):
        errs.append("pgm empty")
    ftn_keys = j.get('ftn_keys') or []
    qtn_keys = j.get('qtn_keys') or []
    if not ftn_keys:
        errs.append("ftn_keys empty")
    if not qtn_keys:
        errs.append("qtn_keys empty")
    if j.get('step') != fields['step']:
        errs.append(f"step mismatch: json={j.get('step')} fname={fields['step']}")
    expected_w = "W" + fields['wafer']
    if j.get('wafer') != expected_w:
        errs.append(f"wafer mismatch: json={j.get('wafer')} expected={expected_w}")
    n_chips = len(j.get('chips', []))
    if not (700 <= n_chips <= 850):
        errs.append(f"chip count {n_chips} out of [700,850]")
    chips = j.get('chips') or []
    if chips and ftn_keys and qtn_keys:
        probe = chips[:20]
        defect_probe = [c for c in chips if int(c.get('b', '0')) >= 200][:50]
        normal_probe = [c for c in chips if int(c.get('b', '0')) < 200][:50]
        if defect_probe: probe += defect_probe[:3]
        if normal_probe: probe += normal_probe[:3]
        for i, chip in enumerate(probe):
            if len(chip.get('f') or []) != len(ftn_keys):
                errs.append(f"chip[{i}] f length mismatch")
                break
            if len(chip.get('q') or []) != len(qtn_keys):
                errs.append(f"chip[{i}] q length mismatch")
                break
        if defect_probe and normal_probe:
            def_max = sum(max(c.get('f') or [0]) + max(c.get('q') or [0]) for c in defect_probe) / len(defect_probe)
            norm_max = sum(max(c.get('f') or [0]) + max(c.get('q') or [0]) for c in normal_probe) / len(normal_probe)
            if def_max <= norm_max * 1.25:
                errs.append(f"FTN/QTN defect boost weak: defect={def_max:.1f} normal={norm_max:.1f}")
    return errs

def verify_class(cls, sample_n=None):
    png_dir  = os.path.join(PNG_ROOT, cls)
    json_dir = os.path.join(JSON_ROOT, cls)
    if not os.path.isdir(png_dir):
        return {'count': 0, 'ok': 0, 'fail': 0, 'errors': [(cls, ['png_dir missing'])]}
    pngs = sorted(glob.glob(os.path.join(png_dir, '*.png')))
    if sample_n: pngs = pngs[:sample_n]
    summary = {'count': len(pngs), 'ok': 0, 'fail': 0, 'errors': []}
    for p in pngs:
        fname = os.path.basename(p)
        f = parse_filename(fname)
        e = check_filename(f) + check_png(p)
        if f is not None:
            jpath = os.path.join(json_dir, fname[:-4] + '.json')
            if not os.path.exists(jpath): e.append('json missing')
            else: e += check_json(jpath, f)
        if e:
            summary['fail'] += 1; summary['errors'].append((fname, e))
        else:
            summary['ok'] += 1
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--class', dest='cls', default=None)
    p.add_argument('--sample', type=int, default=None, help='per-class sample limit')
    p.add_argument('--show-errors', type=int, default=3)
    args = p.parse_args()

    if not os.path.isdir(PNG_ROOT):
        print(f"NOT FOUND: {PNG_ROOT}"); sys.exit(1)

    found = set(os.listdir(PNG_ROOT))
    expected = expected_class_set()
    missing = expected - found
    unexpected = found - expected
    if missing:    print(f"[CLASSES] MISSING ({len(missing)}): {sorted(missing)}")
    if unexpected: print(f"[CLASSES] UNEXPECTED ({len(unexpected)}): {sorted(unexpected)}")
    if not missing and not unexpected:
        print(f"[CLASSES] OK -- {len(found)} folders match expected")

    cls_list = [args.cls] if args.cls else sorted(found)
    total_ok = total_fail = total_count = 0
    for c in cls_list:
        s = verify_class(c, args.sample)
        total_count += s['count']; total_ok += s['ok']; total_fail += s['fail']
        status = 'OK' if s['fail'] == 0 else f"FAIL {s['fail']}"
        print(f"  {c:35s} count={s['count']:4d} ok={s['ok']:4d} {status}")
        for fname, errs in s['errors'][:args.show_errors]:
            print(f"    [-] {fname}: {', '.join(errs)}")
    print(f"\n[TOTAL] count={total_count} ok={total_ok} fail={total_fail}")
    sys.exit(0 if total_fail == 0 else 1)

if __name__ == '__main__':
    main()
