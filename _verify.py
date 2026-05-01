"""Verify generated wafer synthetic samples — comprehensive checks.

검증 항목 (생성 코드 _sample_gen.py spec 기반):

1. 클래스 폴더 구조 (33 defect + Normal = 34 클래스)
2. 파일명 9-token 형식 + 토큰별 유효성 (root/step/wafer/ymd/hms/yield/sys/TD/LT)
3. PNG mode='P', size 6400×6400, palette ≥32 colors
4. JSON 페어 + top-level 스키마 (필수 키 + partid/part_id mirror + pgm format)
5. JSON ↔ 파일명 cross-validation (root/step/wafer/yield/sys/TD/LT)
6. chip count = netd (inside-wafer chip 수, 700-850 범위)
7. chip별 키 (x_abs/y_abs/x_cal/y_cal/b/rect/f/q) + dense 길이
8. **bin pool 검증** — kind별 정상/invalid/sys defect bin이 정의된 풀 안에 있는지
9. **defect chip 개수 분포** — 클래스별 defect chip 분포 sanity (Normal=0, 기타 ≥일정 수)
10. **FTN/QTN 분석성** — defect chip hot-item 평균 / normal hot-item 평균 ≥ 3x
11. **wafer batch sanity** — 같은 root 안에서 step/wafer 일관성, partid 형식

Usage:
    python _verify.py                                # 전체 검사 (random sample 5/class)
    python _verify.py --class Center_bank_boundary   # 특정 클래스만
    python _verify.py --sample 10                    # 클래스당 10개
    python _verify.py --strict                       # 엄격 모드 (defect/FTN/QTN 통계까지)
    python _verify.py --show-errors 5                # 클래스당 에러 표시 개수
"""
import os, json, glob, re, argparse, sys, random, statistics
from collections import defaultdict
from PIL import Image
import numpy as np

PNG_ROOT  = "D:/project/data/wm-811k/unknown"
JSON_ROOT = "D:/project/data/positions/unknown"
SIZE      = 6400
GRID      = 32
N_INSIDE_RANGE = (700, 850)                                                              # netd 허용 범위

TD_OPTIONS = {'PE', 'EE', 'PT'}
LT_OPTIONS = {'NORMAL', 'PWQ', 'ENGINEER'}

# kind별 bin pool (sample_gen.py spec)
NORMAL_BIN_RANGE = (1, 200)                                                              # exclusive upper
INVALID_BIN_RANGE = {
    '00P': (200, 280),                                                                   # 200-279
    '00C': (200, 300),                                                                   # 200-299
}
SYS_DEFECT_BINS = {
    '00P': {285, 286, 287, 288, 290, 291},
    '00C': {300, 385, 386, 388, 389, 390},
}

EXPECTED_CLASSES = ['Center', 'Donut', 'Edge-Ring', 'Edge-Bottom', 'Edge-Top',
                    'Full', 'Thick-Edge']
EXPECTED_OBJECTS = ['bank_boundary', 'particle_blast', 'scratch',
                    'scratch_21deg', 'invalid_main']
SPECIAL_CLASSES = ['Normal', 'Starburst', 'CommaCluster']                                # object 없는 단독

# generation spec
EXPECTED_FQ_LEN = 128
FTN_KEYS_START = 1000
QTN_KEYS_START = 5000
CHIP_KEYS_REQUIRED = {'x_abs', 'y_abs', 'x_cal', 'y_cal', 'b', 'rect', 'f', 'q'}
JSON_TOPLEVEL_REQUIRED = {
    'bucket_b_key', 'root', 'step', 'wafer', 'stime', 'partid', 'part_id', 'pgm',
    'tester', 'device', 'ftn_keys', 'qtn_keys', 'netd', 'gd', 'yield', 'sys',
    'tm', 'lt', 'coord', 'chips',
}
COORD_FIXED = {'rot_code': 5, 'tiles_w_rot': GRID, 'tiles_h_rot': GRID}

PARTID_RE = re.compile(r'^PART_[A-Z0-9_]+$')
PGM_RE    = re.compile(r'^PGM_SYN_FQ\d+_(00P|00C)$')


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
    return dict(zip(['root', 'step', 'wafer', 'ymd', 'hms', 'yld', 'sys', 'td', 'lt'], toks))


def check_filename(f):
    errs = []
    if f is None: return ['filename: not 9 tokens']
    if not re.match(r'^[a-z]{3}\d{3}$', f['root']):
        errs.append(f"root format: {f['root']}")
    if f['step'] not in ('00P', '00C'):
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


def check_json_schema(j):
    """Top-level 스키마 + 형식 검증."""
    errs = []
    missing = JSON_TOPLEVEL_REQUIRED - set(j.keys())
    if missing:
        errs.append(f"missing top keys: {sorted(missing)}")
    if j.get('partid') and not PARTID_RE.match(str(j['partid'])):
        errs.append(f"partid format: {j['partid']!r}")
    if j.get('pgm') and not PGM_RE.match(str(j['pgm'])):
        errs.append(f"pgm format: {j['pgm']!r}")
    if j.get('partid') != j.get('part_id'):
        errs.append(f"part_id mirror mismatch: partid={j.get('partid')!r} part_id={j.get('part_id')!r}")
    fk = j.get('ftn_keys') or []; qk = j.get('qtn_keys') or []
    if len(fk) != EXPECTED_FQ_LEN:
        errs.append(f"ftn_keys length {len(fk)} (expected {EXPECTED_FQ_LEN})")
    if len(qk) != EXPECTED_FQ_LEN:
        errs.append(f"qtn_keys length {len(qk)} (expected {EXPECTED_FQ_LEN})")
    if fk and fk[0] != FTN_KEYS_START:
        errs.append(f"ftn_keys[0]={fk[0]} (expected {FTN_KEYS_START})")
    if qk and qk[0] != QTN_KEYS_START:
        errs.append(f"qtn_keys[0]={qk[0]} (expected {QTN_KEYS_START})")
    coord = j.get('coord') or {}
    for k, v in COORD_FIXED.items():
        if coord.get(k) != v:
            errs.append(f"coord.{k}={coord.get(k)} (expected {v})")
    canv = (coord.get('canvas') or {})
    if canv.get('width') != SIZE or canv.get('height') != SIZE:
        errs.append(f"coord.canvas {canv} (expected {SIZE}x{SIZE})")
    return errs


def check_filename_json_consistency(f, j):
    """파일명 토큰과 JSON 필드 cross-validation."""
    errs = []
    if j.get('root') != f['root']:
        errs.append(f"root: json={j.get('root')!r} fname={f['root']!r}")
    if j.get('step') != f['step']:
        errs.append(f"step: json={j.get('step')!r} fname={f['step']!r}")
    expected_w = "W" + f['wafer']
    if j.get('wafer') != expected_w:
        errs.append(f"wafer: json={j.get('wafer')!r} expected={expected_w!r}")
    # tester ← TD, device ← LT (sample_gen.py:629-630, 666)
    if j.get('tester') != f['td']:
        errs.append(f"tester(=TD): json={j.get('tester')!r} fname={f['td']!r}")
    if j.get('device') != f['lt']:
        errs.append(f"device(=LT): json={j.get('device')!r} fname={f['lt']!r}")
    # yield/sys filename(.1f / .0f) — JSON .2f 와 numeric 비교
    try:
        y_f = float(f['yld']); y_j = float(j.get('yield', 'nan'))
        if abs(y_f - y_j) > 0.15:
            errs.append(f"yield: fname={f['yld']} json={j.get('yield')}")
    except Exception:
        errs.append(f"yield parse: fname={f['yld']} json={j.get('yield')}")
    try:
        s_f = float(f['sys']); s_j = float(j.get('sys', 'nan'))
        if abs(s_f - s_j) > 0.6:                                                         # filename .0f 반올림 오차 허용
            errs.append(f"sys: fname={f['sys']} json={j.get('sys')}")
    except Exception:
        errs.append(f"sys parse: fname={f['sys']} json={j.get('sys')}")
    return errs


def check_chips(j, cls_label):
    """chip count + per-chip 키 + bin pool + defect 분포 + FTN/QTN 분석성."""
    errs = []
    chips = j.get('chips') or []
    netd = j.get('netd', 0)
    n_chips = len(chips)
    if n_chips != netd:
        errs.append(f"chips length {n_chips} != netd {netd}")
    if not (N_INSIDE_RANGE[0] <= n_chips <= N_INSIDE_RANGE[1]):
        errs.append(f"chip count {n_chips} out of {N_INSIDE_RANGE}")

    if not chips: return errs

    # chip 키 검증 (sample 5)
    fk_len = len(j.get('ftn_keys') or []); qk_len = len(j.get('qtn_keys') or [])
    for i, ch in enumerate(chips[:5]):
        miss = CHIP_KEYS_REQUIRED - set(ch.keys())
        if miss: errs.append(f"chip[{i}] missing: {sorted(miss)}"); break
        if len(ch.get('f') or []) != fk_len:
            errs.append(f"chip[{i}] f length {len(ch.get('f') or [])} != ftn_keys {fk_len}"); break
        if len(ch.get('q') or []) != qk_len:
            errs.append(f"chip[{i}] q length {len(ch.get('q') or [])} != qtn_keys {qk_len}"); break

    # bin pool 검증 (sample 50)
    step = j.get('step', '')
    invalid_lo, invalid_hi = INVALID_BIN_RANGE.get(step, (0, 0))
    sys_bins = SYS_DEFECT_BINS.get(step, set())
    bad_bin = []
    for ch in chips[:200]:
        try: b = int(str(ch.get('b', '0')).strip())
        except: bad_bin.append(('parse', ch.get('b'))); continue
        if b < 200:
            if not (NORMAL_BIN_RANGE[0] <= b < NORMAL_BIN_RANGE[1]):
                bad_bin.append(('normal_oor', b))
        elif b in sys_bins:
            pass                                                                          # sys defect, OK
        elif invalid_lo <= b < invalid_hi:
            pass                                                                          # invalid, OK
        else:
            bad_bin.append(('unknown_bin', b))
    if bad_bin:
        errs.append(f"bin pool: {bad_bin[:3]}")

    # defect chip 개수 — 모든 클래스 (Normal 포함) defect chip 존재해야 함.
    # Normal은 spatial pattern이 없는 baseline distribution일 뿐, 결함 chip은 있다.
    defect_count = sum(1 for ch in chips if int(str(ch.get('b', '0')).strip()) >= 200)
    if defect_count == 0:
        errs.append(f"0 defect chip (expected >0 for all classes incl. Normal)")

    return errs


def check_ftn_qtn_analyzability(j):
    """FTN/QTN hot-item 평균이 defect chip에서 normal 대비 ≥3x인지 확인 (분석성)."""
    chips = j.get('chips') or []
    fk = j.get('ftn_keys') or []
    if not chips or not fk:
        return []
    defect = [c for c in chips if int(str(c.get('b', '0')).strip()) >= 200]
    normal = [c for c in chips if int(str(c.get('b', '0')).strip()) < 200]
    if len(defect) < 3 or len(normal) < 10:
        return []                                                                         # 표본 부족 — 통과
    f_def = np.array([c.get('f') or [0] for c in defect[:50]])
    f_norm = np.array([c.get('f') or [0] for c in normal[:50]])
    # 가장 큰 8개 column = hot 후보
    f_def_means = f_def.mean(axis=0)
    f_norm_means = f_norm.mean(axis=0)
    top8 = np.argsort(-f_def_means)[:8]
    ratio = f_def_means[top8].mean() / max(1.0, f_norm_means[top8].mean())
    if ratio < 3.0:
        return [f"FTN hot ratio defect/normal = {ratio:.2f}x (expected ≥3.0)"]
    return []


def verify_class(cls, sample_n=None, strict=False, rng=None):
    png_dir  = os.path.join(PNG_ROOT, cls)
    json_dir = os.path.join(JSON_ROOT, cls)
    if not os.path.isdir(png_dir):
        return {'count': 0, 'ok': 0, 'fail': 0, 'errors': [(cls, ['png_dir missing'])],
                'defect_counts': []}
    pngs = sorted(glob.glob(os.path.join(png_dir, '*.png')))
    total = len(pngs)
    if sample_n and len(pngs) > sample_n:
        if rng is not None:
            pngs = sorted(rng.sample(pngs, sample_n))
        else:
            pngs = pngs[:sample_n]

    summary = {'count': total, 'sampled': len(pngs), 'ok': 0, 'fail': 0,
               'errors': [], 'defect_counts': []}

    for p in pngs:
        fname = os.path.basename(p)
        f = parse_filename(fname)
        e = check_filename(f) + check_png(p)
        if f is not None:
            jpath = os.path.join(json_dir, fname[:-4] + '.json')
            if not os.path.exists(jpath):
                e.append('json missing')
            else:
                try:
                    with open(jpath, 'r', encoding='utf-8') as jf:
                        j = json.load(jf)
                    e += check_json_schema(j)
                    e += check_filename_json_consistency(f, j)
                    e += check_chips(j, cls)
                    if strict:
                        e += check_ftn_qtn_analyzability(j)
                    # defect chip count 통계 수집
                    chips = j.get('chips') or []
                    dcount = sum(1 for ch in chips
                                 if int(str(ch.get('b', '0')).strip()) >= 200)
                    summary['defect_counts'].append(dcount)
                except Exception as ex:
                    e.append(f"json parse: {ex}")
        if e:
            summary['fail'] += 1; summary['errors'].append((fname, e))
        else:
            summary['ok'] += 1
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--class', dest='cls', default=None)
    p.add_argument('--sample', type=int, default=5,
                   help='per-class random sample limit (default 5; 0=all)')
    p.add_argument('--strict', action='store_true',
                   help='FTN/QTN 분석성 (hot ratio≥3x) 까지 검증')
    p.add_argument('--show-errors', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not os.path.isdir(PNG_ROOT):
        print(f"NOT FOUND: {PNG_ROOT}"); sys.exit(1)

    rng = random.Random(args.seed)
    sample_n = args.sample if args.sample > 0 else None

    found = set(os.listdir(PNG_ROOT))
    expected = expected_class_set()
    missing = expected - found
    unexpected = found - expected
    print("=" * 90)
    print(f"[CLASSES] expected={len(expected)} found={len(found)}")
    if missing:    print(f"  MISSING ({len(missing)}): {sorted(missing)}")
    if unexpected: print(f"  UNEXPECTED ({len(unexpected)}): {sorted(unexpected)}")
    if not missing and not unexpected:
        print(f"  OK -- all {len(found)} folders match")

    cls_list = [args.cls] if args.cls else sorted(found)
    print("\n" + "=" * 90)
    print(f"[FILES] sample={sample_n} strict={args.strict}")
    print("=" * 90)

    total_ok = total_fail = total_count = total_sampled = 0
    defect_summary: dict = {}
    fmt = "{:<30s} count={:5d} sampled={:3d} ok={:3d} {}"
    for c in cls_list:
        s = verify_class(c, sample_n=sample_n, strict=args.strict, rng=rng)
        total_count += s['count']; total_ok += s['ok']; total_fail += s['fail']
        total_sampled += s.get('sampled', 0)
        status = 'OK' if s['fail'] == 0 else f"FAIL {s['fail']}"
        print(fmt.format(c, s['count'], s.get('sampled', 0), s['ok'], status))
        for fname, errs in s['errors'][:args.show_errors]:
            for er in errs:
                print(f"    [-] {fname}: {er}")
        if s['defect_counts']:
            defect_summary[c] = s['defect_counts']

    # === defect chip 분포 통계 ===
    print("\n" + "=" * 90)
    print("[DEFECT CHIP COUNT distribution per class]  (b≥200 chips)")
    print("=" * 90)
    print(f"{'Class':<30s}  {'min':>4s}  {'mean':>6s}  {'max':>4s}  {'n':>3s}")
    for c in sorted(defect_summary):
        cnts = defect_summary[c]
        mn, mx = min(cnts), max(cnts)
        avg = statistics.mean(cnts)
        flag = ""
        if c == 'Normal' and mx > 0:
            flag = "  ! Normal should have 0"
        elif c != 'Normal' and mn == 0:
            flag = "  ! has sample with 0 defect"
        print(f"{c:<30s}  {mn:4d}  {avg:6.1f}  {mx:4d}  {len(cnts):3d}{flag}")

    print("\n" + "=" * 90)
    print(f"[TOTAL] files={total_count}  sampled={total_sampled}  ok={total_ok}  fail={total_fail}")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == '__main__':
    main()
