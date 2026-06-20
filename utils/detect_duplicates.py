'''
detect_duplicates.py — Perceptual duplicate detection for the CMR HDF5 datastore.

SHA256 (generate_checksums.py) only catches byte-identical dataset arrays. It
misses true duplicates that were processed on different servers, because
non-deterministic series ordering (pre-natsort), non-deterministic frame ordering
(fixed only recently) and tiny normalization/interpolation differences make the
same study hash differently.

This tool instead compares the *imagery*. It ports the findimagedupes perceptual
fingerprint (160 -> grey -> blur -> normalize -> equalize -> 16x16 -> threshold
= 256-bit hash, compared by Hamming distance) to pure Python, fingerprints the
first / middle / last frame of every series of every accession (single-frame, non
-cine series are handled too), then finds pairs/clusters of accession files whose
imagery is near-identical and writes a CSV manifest of the flagged
anon_mrn/anon_accession.h5 files for review.

Robust to the frame/series ordering issues above: every series contributes
several frames and all frames across the whole datastore are matched globally, so
ordering does not matter.

Threshold uses the real findimagedupes formula: allowed_bits = floor(2.56 * (100
- pct)). So --threshold 99 allows <=2 of 256 bits to differ (near-identical).

The full fingerprint manifest is always written to disk so future datasets can be
checked against it WITHOUT recomputing — scan a new batch and pass the prior
manifest with --reference-csv to find cross-dataset overlap.

Outputs default to dedup_scan_outs/ created next to this script.

Usage:
    # full scan of a datastore
    python detect_duplicates.py -i /path/to/datastore -c 12 --threshold 99

    # check a NEW batch against a previously saved fingerprint DB (no recompute of the DB)
    python detect_duplicates.py -i /path/to/new_batch -c 12 \
        --reference-csv dedup_scan_outs/jun20_2026_fingerprint_db.csv

    # re-run matching on an existing manifest only (skip fingerprinting)
    python detect_duplicates.py --fingerprint-csv dedup_scan_outs/jun20_2026_dup_fingerprints.csv

Created by Rohan Shad, MD (perceptual dedup utility)
'''

import os
import glob
import math
import time
import argparse as ap
import multiprocessing
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import bcolors

# --- Fingerprint constants (chosen for internal consistency, not ImageMagick bit-fidelity) ---
SAMPLE_SIZE = 160          # findimagedupes: Sample 160x160!
BLUR_RADIUS = 3            # findimagedupes: Blur radius=3 (sigma large -> ~flat 7px kernel == BoxBlur(3))
GRID = 16                  # findimagedupes: Sample 16x16  -> 16*16 = 256-bit fingerprint
FP_BYTES = (GRID * GRID) // 8   # 32 bytes
RESAMPLE = Image.Resampling.BILINEAR
ALL_POSITIONS = ('first', 'middle', 'last')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, 'dedup_scan_outs')
MANIFEST_COLS = ['source', 'relpath', 'mrn', 'accession', 'series', 'frame_pos', 'n_frames', 'fp_hex']

# Vectorized popcount over (P,4) uint64 rows; falls back for numpy < 2.0 (no bitwise_count).
if hasattr(np, 'bitwise_count'):
	def _popcount_rows(xor):
		return np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
else:
	_NIBBLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
	def _popcount_rows(xor):
		return _NIBBLE[xor.view(np.uint8)].sum(axis=1)


# ----------------------------------------------------------------------------- #
#  Stage 1: fingerprinting                                                       #
# ----------------------------------------------------------------------------- #
def fingerprint(frame2d):
	'''
	Compute the 256-bit findimagedupes-style perceptual fingerprint of a single
	2D frame. Returns 32 raw bytes, or None for a degenerate (near-constant)
	frame that cannot be normalized.
	'''
	f = np.asarray(frame2d, dtype=np.float64)
	vmin, vmax = float(f.min()), float(f.max())
	if vmax - vmin < 1e-6:                       # blank / constant -> not hashable
		return None

	g = (((f - vmin) / (vmax - vmin)) * 255.0).astype(np.uint8)
	img = Image.fromarray(g).resize((SAMPLE_SIZE, SAMPLE_SIZE), RESAMPLE)   # 2D uint8 -> mode 'L'
	img = img.filter(ImageFilter.BoxBlur(BLUR_RADIUS))   # heavy low-pass
	img = ImageOps.equalize(ImageOps.autocontrast(img))  # Normalize + Equalize
	a = np.asarray(img.resize((GRID, GRID), RESAMPLE), dtype=np.uint8)

	bits = (a > np.median(a)).flatten()          # threshold at median (== IM 50% post-equalize)
	return np.packbits(bits).tobytes()           # 32 bytes


def _attr_int(dset, key):
	'''Read an integer-valued HDF5 attribute, or None if absent/unparseable.'''
	try:
		v = dset.attrs.get(key)
		if v is None:
			return None
		return int(np.asarray(v).flatten()[0])
	except Exception:
		return None


def series_layout(shape, n_attr):
	'''
	Infer (frame_axis, channel_axis, n_frames) for a series dataset, handling every
	layout in the repo plus single-frame (non-cine) series:
	  (h,w)            -> single 2D frame
	  (f,h,w)          -> grey cine            (f==1 is a valid single-frame cine)
	  (c,h,w)          -> single-frame, c channels
	  (f,c,h,w)        -> new layout cine
	  (c,f,h,w)        -> old layout cine
	"rgb" is grey repeated across channels, so taking channel 0 is lossless.
	frame_axis is None for a single 2D frame; n_frames is 1 for single-frame series.

	The channel axis is the one of size 1 or 3; total_images (n_attr), when present,
	authoritatively fixes the frame axis. When both leading 4D axes are in {1,3}
	(ambiguous) a singleton axis is taken as the (single) frame axis.
	'''
	nd = len(shape)
	if nd == 2:                                  # (h,w) single frame
		return (None, None, 1)
	if nd == 3:
		a0 = shape[0]
		if a0 in (1, 3) and not (n_attr is not None and n_attr == a0):
			return (None, 0, 1)                  # (c,h,w) single frame
		return (0, None, a0)                     # (f,h,w) grey cine (incl. f==1)
	if nd == 4:
		a0, a1 = shape[0], shape[1]
		if n_attr is not None and a0 == n_attr and a1 != n_attr:
			return (0, 1, a0)                    # attr-confirmed (f,c,h,w)
		if n_attr is not None and a1 == n_attr and a0 != n_attr:
			return (1, 0, a1)                    # attr-confirmed (c,f,h,w)
		if a1 in (1, 3) and a0 not in (1, 3):
			return (0, 1, a0)                    # (f,c,h,w) new
		if a0 in (1, 3) and a1 not in (1, 3):
			return (1, 0, a1)                    # (c,f,h,w) old
		# both leading axes in {1,3}: a singleton axis is the (single) frame axis
		if a0 == 1 and a1 != 1:
			return (0, 1, a0)                    # (1,c,h,w) single frame, new layout
		if a1 == 1 and a0 != 1:
			return (1, 0, a1)                    # (c,1,h,w) single frame, old layout
		return (0, 1, a0)                        # fully ambiguous (1x1 / 3x3) -> assume new
	raise ValueError(f'unsupported dataset ndim={nd}')


def read_plane(dset, layout, fi):
	'''Read a single frame index as a 2D (h,w) array without loading the whole cine.'''
	frame_axis, _channel_axis, _ = layout
	nd = dset.ndim
	if nd == 2:
		return dset[()]
	if nd == 3:
		if frame_axis == 0:
			return dset[fi]                      # grey cine
		return dset[0]                           # channels, single frame -> channel 0
	# nd == 4
	if frame_axis == 0:
		return dset[fi, 0]                       # (f,c,h,w)
	return dset[0, fi]                           # (c,f,h,w)


def fingerprint_series(dset, positions):
	'''Yield (position, n_frames, fingerprint_bytes_or_None) for the requested frames.'''
	layout = series_layout(dset.shape, _attr_int(dset, 'total_images'))
	n = layout[2]
	wanted = {'first': 0, 'middle': n // 2, 'last': n - 1}
	seen = set()
	out = []
	for pos in positions:
		fi = wanted[pos]
		if fi in seen:                           # collapse when n < 3 (incl. single frame)
			continue
		seen.add(fi)
		plane = np.asarray(read_plane(dset, layout, fi))
		out.append((pos, n, fingerprint(plane)))
	return out


def fingerprint_file(path, input_dir, positions, source):
	'''
	Worker: open one .h5 accession file and fingerprint the requested frames of
	every series. Returns a list of rows matching MANIFEST_COLS. Degenerate frames
	get an empty fp_hex (kept for transparency, dropped later).
	'''
	relpath = os.path.relpath(path, input_dir)
	mrn = os.path.split(os.path.dirname(path))[1]
	accession = os.path.basename(path)[:-3]
	rows = []
	try:
		with h5py.File(path, 'r') as f:
			for series in list(f.keys()):
				try:
					dset = f[series]
					if not isinstance(dset, h5py.Dataset):
						continue
					for pos, n, fp in fingerprint_series(dset, positions):
						rows.append([source, relpath, mrn, accession, series, pos, n,
						             fp.hex() if fp is not None else ''])
				except Exception as ex:
					print(f'WARN: {relpath}::{series} skipped -> {ex}')
	except Exception as ex:
		print(f'{bcolors.FAIL}FAIL{bcolors.ENDC}: {relpath} -> {ex}')
		return []
	return rows


def build_fingerprint_manifest(input_dir, cpus, positions, institution_prefix, source):
	'''Walk the datastore (anon_mrn/anon_accession.h5) and fingerprint everything.'''
	if institution_prefix:
		filelist = glob.glob(os.path.join(input_dir, f'{institution_prefix}*', '*h5'))
	else:
		filelist = glob.glob(os.path.join(input_dir, '*', '*h5'))
	filelist.sort()
	print(f'Found {len(filelist)} accession files under {input_dir}  (source label: {source})')

	rows = []
	if cpus > 1:
		p = multiprocessing.Pool(processes=cpus)
		async_results = [p.apply_async(fingerprint_file, [fp, input_dir, positions, source]) for fp in filelist]
		p.close()
		p.join()
		for k, r in enumerate(async_results):
			rows.extend(r.get())
			if (k + 1) % 500 == 0:
				print(f'  ...fingerprinted {k + 1}/{len(filelist)} files')
	else:
		for k, fp in enumerate(filelist):
			rows.extend(fingerprint_file(fp, input_dir, positions, source))
			if (k + 1) % 500 == 0:
				print(f'  ...fingerprinted {k + 1}/{len(filelist)} files')

	return pd.DataFrame(rows, columns=MANIFEST_COLS)


def load_manifest(path, default_label):
	'''Load a saved fingerprint manifest, tolerating older schemas (no source column).'''
	df = pd.read_csv(path, dtype={'fp_hex': str})
	if 'source' not in df.columns:
		df['source'] = default_label
	for col in MANIFEST_COLS:
		if col not in df.columns:
			df[col] = 0 if col == 'n_frames' else ''
	return df[MANIFEST_COLS]


# ----------------------------------------------------------------------------- #
#  Stage 2: near-duplicate search + aggregation                                 #
# ----------------------------------------------------------------------------- #
def load_codes(df):
	'''
	Keep only rows with a valid 64-hex fingerprint; return (clean_df, fp_bytes
	list, codes uint64[N,4]). Drops degenerate/empty fingerprints.
	'''
	mask = df['fp_hex'].astype(str).str.fullmatch(r'[0-9a-fA-F]{%d}' % (FP_BYTES * 2))
	clean = df[mask.fillna(False)].reset_index(drop=True)
	fp_bytes = [bytes.fromhex(h) for h in clean['fp_hex']]
	codes = np.zeros((len(fp_bytes), 4), dtype=np.uint64)
	for i, b in enumerate(fp_bytes):
		codes[i] = np.frombuffer(b, dtype='>u8')
	return clean, fp_bytes, codes


def candidate_pairs(fp_bytes, n_bands, max_bucket):
	'''
	Multi-index hashing (pigeonhole): split each 256-bit code into n_bands
	byte-boundary segments. Two codes within Hamming d agree on >=1 band when
	n_bands = d + 1, so any near-duplicate pair shares a bucket. Returns the set
	of candidate (i,j) index pairs (i<j) plus a count of oversized buckets skipped.
	'''
	N = len(fp_bytes)
	bounds = [round(k * FP_BYTES / n_bands) for k in range(n_bands + 1)]
	pairs = set()
	skipped = 0
	for k in range(n_bands):
		s, e = bounds[k], bounds[k + 1]
		if s == e:
			continue
		buckets = defaultdict(list)
		for i in range(N):
			buckets[fp_bytes[i][s:e]].append(i)
		for members in buckets.values():
			m = len(members)
			if m < 2:
				continue
			if m > max_bucket:
				skipped += 1
				print(f'WARN: band bucket of {m} codes exceeds --max-bucket={max_bucket}; '
				      f'skipping (likely a large genuine duplicate set — review manually)')
				continue
			for a in range(m):
				for b in range(a + 1, m):
					ia, ib = members[a], members[b]
					pairs.add((ia, ib) if ia < ib else (ib, ia))
	return pairs, skipped


def verify_pairs(pairs, codes, fids, max_bits):
	'''Exact-Hamming verification of candidate pairs; keep cross-file pairs within max_bits.'''
	if not pairs:
		return []
	I = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
	J = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
	xor = codes[I] ^ codes[J]
	dist = _popcount_rows(xor)
	keep = dist <= max_bits
	out = []
	for i, j, d in zip(I[keep].tolist(), J[keep].tolist(), dist[keep].tolist()):
		if fids[i] != fids[j]:                   # never a duplicate of itself (source+relpath)
			out.append((i, j, int(d)))
	return out


class UnionFind:
	def __init__(self):
		self.parent = {}

	def find(self, x):
		self.parent.setdefault(x, x)
		root = x
		while self.parent[root] != root:
			root = self.parent[root]
		while self.parent[x] != root:            # path compression
			self.parent[x], x = root, self.parent[x]
		return root

	def union(self, a, b):
		ra, rb = self.find(a), self.find(b)
		if ra != rb:
			self.parent[ra] = rb


def aggregate_to_files(frame_pairs, df, min_series, new_sources):
	'''
	Roll frame-level matches up to file pairs, cluster them, and return
	(pair_rows, flagged_rows, n_clusters). A file is identified by (source, relpath).
	A file pair is flagged when it shares at least min_series near-identical series.
	When new_sources is not None, pairs whose BOTH files are reference-only are
	dropped (focus the report on the freshly scanned data).
	'''
	src = df['source'].tolist()
	rel = df['relpath'].tolist()
	ser = df['series'].tolist()
	n_series = df.groupby(['source', 'relpath'])['series'].nunique().to_dict()

	agg = {}
	for i, j, d in frame_pairs:
		ka, sa = (src[i], rel[i]), ser[i]
		kb, sb = (src[j], rel[j]), ser[j]
		if ka == kb:
			continue
		if new_sources is not None and ka[0] not in new_sources and kb[0] not in new_sources:
			continue
		(fa, sea), (fb, seb) = sorted([(ka, sa), (kb, sb)])
		rec = agg.get((fa, fb))
		if rec is None:
			rec = {'series_a': set(), 'series_b': set(), 'frames': 0,
			       'min_h': d, 'ex_a': sea, 'ex_b': seb}
			agg[(fa, fb)] = rec
		rec['series_a'].add(sea)
		rec['series_b'].add(seb)
		rec['frames'] += 1
		if d < rec['min_h']:
			rec['min_h'], rec['ex_a'], rec['ex_b'] = d, sea, seb

	kept = []
	for (fa, fb), rec in agg.items():
		matching = len(rec['series_a'])
		if matching < min_series:
			continue
		na, nb = n_series.get(fa, 0), n_series.get(fb, 0)
		frac = round(matching / max(1, min(na, nb)), 4)
		kept.append((fa, fb, matching, na, nb, frac, rec['min_h'],
		             rec['frames'], rec['ex_a'], rec['ex_b']))

	# cluster connected components of the file-pair graph
	uf = UnionFind()
	partners = defaultdict(set)
	for fa, fb, *_ in kept:
		uf.union(fa, fb)
		partners[fa].add(fb)
		partners[fb].add(fa)

	cluster_id = {}
	next_id = 1
	for f in sorted(partners):
		root = uf.find(f)
		if root not in cluster_id:
			cluster_id[root] = next_id
			next_id += 1

	pair_rows = []
	for fa, fb, matching, na, nb, frac, minh, frames, exa, exb in kept:
		pair_rows.append([cluster_id[uf.find(fa)], fa[0], fa[1], fb[0], fb[1],
		                  matching, na, nb, frac, minh, frames, exa, exb])
	pair_rows.sort(key=lambda r: (-r[8], r[9]))      # match_fraction desc, min_hamming asc

	flagged_rows = [[f[0], f[1], cluster_id[uf.find(f)], len(partners[f])] for f in sorted(partners)]
	flagged_rows.sort(key=lambda r: (r[2], r[0], r[1]))
	return pair_rows, flagged_rows, (next_id - 1)


# ----------------------------------------------------------------------------- #
#  Main                                                                          #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
	parser = ap.ArgumentParser(
		description="Perceptual (findimagedupes-style) duplicate detection for the CMR HDF5 datastore.",
		epilog="Created by Rohan Shad, MD")
	parser.add_argument('-i', '--input_dir', default=None,
	                    help='Datastore root containing anon_mrn/anon_accession.h5 (omit only with --fingerprint-csv)')
	parser.add_argument('-o', '--output_dir', default=None,
	                    help='Output directory (default: dedup_scan_outs/ next to this script)')
	parser.add_argument('-c', '--cpus', type=int, default=12, help='Number of CPUs for fingerprinting')
	parser.add_argument('--threshold', type=float, default=99.0,
	                    help='Similarity %% (findimagedupes); allowed_bits = floor(2.56*(100-thr)). 99 -> <=2 bits')
	parser.add_argument('--frames', default='first,middle,last',
	                    help='Comma list of frames per series to fingerprint (subset of first,middle,last)')
	parser.add_argument('--min-series', type=int, default=1,
	                    help='Min matching series for a file pair to be flagged (raise to cut single-view false positives)')
	parser.add_argument('--max-bucket', type=int, default=5000,
	                    help='Skip+log band buckets larger than this (guards against pathological blowups)')
	parser.add_argument('--fingerprint-csv', default=None,
	                    help='Load this manifest as the PRIMARY set and skip Stage 1 if it exists; else write it here')
	parser.add_argument('--reference-csv', default=None,
	                    help='Comma list of previously saved fingerprint manifests to match against (not recomputed)')
	parser.add_argument('--source-label', default=None,
	                    help='Label for the scanned dataset in the manifest (default: input dir basename)')
	parser.add_argument('--institution_prefix', default=None, help='Only scan anon_mrn dirs starting with this prefix')
	args = parser.parse_args()

	start = time.time()
	output_dir = args.output_dir or DEFAULT_OUTDIR
	os.makedirs(output_dir, exist_ok=True)
	date = time.strftime('%b%d_%Y').lower()

	positions = [pos.strip() for pos in args.frames.split(',') if pos.strip()]
	bad = [pos for pos in positions if pos not in ALL_POSITIONS]
	if bad or not positions:
		raise SystemExit(f'--frames must be a non-empty subset of {ALL_POSITIONS}; got {args.frames!r}')

	max_bits = max(0, math.floor(2.56 * (100.0 - args.threshold)))
	n_bands = max(1, min(FP_BYTES, max_bits + 1))     # pigeonhole; 1 band == exact match (max_bits 0)

	print('------------------------------------')
	print(f'{bcolors.BLUE}Perceptual duplicate detection{bcolors.ENDC}')
	print(f'threshold={args.threshold}%  -> allowed Hamming bits <= {max_bits} (of 256), bands={n_bands}')
	print(f'frames={positions}  min_series={args.min_series}  output={output_dir}')
	print('------------------------------------')

	# Stage 1 — PRIMARY fingerprint manifest (compute fresh, or load to skip fingerprinting)
	fp_manifest_path = args.fingerprint_csv or os.path.join(output_dir, f'{date}_dup_fingerprints.csv')
	if args.fingerprint_csv and os.path.exists(args.fingerprint_csv):
		print(f'Loading PRIMARY fingerprint manifest (skipping Stage 1): {args.fingerprint_csv}')
		default_label = args.source_label or os.path.splitext(os.path.basename(args.fingerprint_csv))[0]
		primary = load_manifest(args.fingerprint_csv, default_label)
		if args.source_label:
			primary['source'] = args.source_label
	else:
		if not args.input_dir:
			raise SystemExit('--input_dir is required unless --fingerprint-csv points at an existing manifest')
		source_label = args.source_label or os.path.basename(os.path.normpath(args.input_dir)) or 'scan'
		primary = build_fingerprint_manifest(args.input_dir, args.cpus, positions, args.institution_prefix, source_label)
		primary.to_csv(fp_manifest_path, index=False)
		print(f'Fingerprint manifest written: {fp_manifest_path}')

	new_sources = set(primary['source'].unique())

	# Optional REFERENCE manifests (matched against, never recomputed)
	ref_frames = []
	if args.reference_csv:
		for ref_path in [p.strip() for p in args.reference_csv.split(',') if p.strip()]:
			if not os.path.exists(ref_path):
				raise SystemExit(f'--reference-csv not found: {ref_path}')
			ref_df = load_manifest(ref_path, os.path.splitext(os.path.basename(ref_path))[0])
			print(f'Reference manifest: {ref_path}  ({len(ref_df)} rows, '
			      f'sources={sorted(ref_df["source"].unique())})')
			ref_frames.append(ref_df)

	df = pd.concat([primary] + ref_frames, ignore_index=True) if ref_frames else primary
	use_new_filter = None if not ref_frames else new_sources   # only restrict to new data when a reference is used

	# Persist a combined DB snapshot so the union can be referenced next time without recompute
	if ref_frames:
		db_path = os.path.join(output_dir, f'{date}_fingerprint_db.csv')
		df.drop_duplicates(subset=['source', 'relpath', 'series', 'frame_pos']).to_csv(db_path, index=False)
		print(f'Combined fingerprint DB written (pass as --reference-csv next time): {db_path}')

	# Stage 2 — near-duplicate search
	clean, fp_bytes, codes = load_codes(df)
	n_total, n_valid = len(df), len(clean)
	n_degenerate = n_total - n_valid
	n_files = clean[['source', 'relpath']].drop_duplicates().shape[0]
	n_series_total = int(clean.groupby(['source', 'relpath'])['series'].nunique().sum())
	print(f'Fingerprints: {n_valid} valid frames across {n_files} files '
	      f'({n_series_total} series); {n_degenerate} degenerate frames skipped')

	fids = list(zip(clean['source'].tolist(), clean['relpath'].tolist()))
	pairs, skipped_buckets = candidate_pairs(fp_bytes, n_bands, args.max_bucket)
	frame_pairs = verify_pairs(pairs, codes, fids, max_bits)
	pair_rows, flagged_rows, n_clusters = aggregate_to_files(frame_pairs, clean, args.min_series, use_new_filter)

	# Outputs
	dup_path = os.path.join(output_dir, f'{date}_duplicate_manifest.csv')
	flag_path = os.path.join(output_dir, f'{date}_flagged_files.csv')
	pd.DataFrame(pair_rows, columns=[
		'cluster_id', 'source_a', 'file_a', 'source_b', 'file_b', 'matching_series',
		'n_series_a', 'n_series_b', 'match_fraction', 'min_hamming', 'matching_frames',
		'example_series_a', 'example_series_b']
	).to_csv(dup_path, index=False)
	pd.DataFrame(flagged_rows, columns=['source', 'relpath', 'cluster_id', 'n_partners']).to_csv(flag_path, index=False)

	elapsed = round(time.time() - start, 2)
	log_path = os.path.join(output_dir, f'{date}_dedup_run.log')
	with open(log_path, 'a') as log:
		log.write(f'# perceptual dedup run — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
		log.write(f'# input:     {args.input_dir}\n')
		log.write(f'# output:    {output_dir}\n')
		log.write(f'# reference: {args.reference_csv}\n')
		log.write(f'# params:    threshold={args.threshold}% allowed_bits<={max_bits} bands={n_bands} '
		          f'frames={",".join(positions)} min_series={args.min_series} max_bucket={args.max_bucket}\n')
		log.write(f'files={n_files} series={n_series_total} valid_frames={n_valid} degenerate={n_degenerate} '
		          f'candidate_pairs={len(pairs)} confirmed_frame_pairs={len(frame_pairs)} '
		          f'flagged_file_pairs={len(pair_rows)} clusters={n_clusters} '
		          f'oversized_buckets_skipped={skipped_buckets} elapsed={elapsed}s\n')

	print('------------------------------------')
	print(f'{bcolors.BLUE}Flagged file pairs:{bcolors.ENDC} {len(pair_rows)}  '
	      f'across {n_clusters} clusters  ({len(flagged_rows)} distinct files)')
	print(f'candidate pairs={len(pairs)}  confirmed frame pairs={len(frame_pairs)}  '
	      f'oversized buckets skipped={skipped_buckets}')
	print(f'Duplicate manifest: {dup_path}')
	print(f'Flagged files:      {flag_path}')
	print(f'Run log:            {log_path}')
	print(f'Elapsed time: {elapsed}s')
	print('------------------------------------')
