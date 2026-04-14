#!/usr/bin/env python3
'''
crosswalk_gen.py — batch-safe anonymization crosswalk generator.

Generates anon_mrn (patient-level) and anon_accession (scan-level) UUIDs for
a new batch of data, while:
  1. Preserving anon_mrn identity for patients seen in prior batches
  2. Detecting and regenerating any UUID collisions with prior data
  3. Warning (not halting) on duplicate scan identifiers across batches

Outputs written to --output-dir:
  {batch_label}_batch_anon.csv     new batch only,  [mrn_col, acc_col, anon_mrn, anon_accession]
  {batch_label}_concat_full.csv    all batches,     all columns
  {batch_label}_concat_anon.csv    all batches,     [mrn_col, acc_col, anon_mrn, anon_accession]

Usage (first batch):
  python crosswalk_gen.py --input batch1.csv --mrn-col "Patient ID" --acc-col "Study ID" \\
	  --batch-label batch_1 --output-dir ./out

Usage (subsequent batch):
  python crosswalk_gen.py --input batch2.csv --prior ./out/batch_1_concat_full.csv \\
	  --mrn-col "Patient ID" --acc-col "Study ID" --batch-label batch_2 --output-dir ./out
  # --prior is always the {prev_batch_label}_concat_full.csv written by the previous run.
'''

import argparse
import uuid
import warnings
from pathlib import Path
import pandas as pd



###### UUID generation ######

def generate_unique_id(existing: set, length: int, max_attempts: int = 1000) -> str:
	'''
	Generate a UUID hex string of `length` chars not already in `existing`.
	Adds the result to `existing` in-place before returning.
	Raises RuntimeError if a unique ID cannot be found within max_attempts.
	'''
	for _ in range(max_attempts):
		candidate = uuid.uuid4().hex[:length].lower()
		if candidate not in existing:
			existing.add(candidate)
			return candidate
	raise RuntimeError(
		f"Could not generate a unique ID after {max_attempts} attempts "
		f"(length={length}). Consider increasing --mrn-len / --acc-len."
	)


###### Validation ######

def _validate_columns(df: pd.DataFrame, required: list, label: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(
			f"Missing columns in {label}: {missing}\n"
			f"Found columns: {list(df.columns)}"
		)


def _sanity_check(concat_df: pd.DataFrame, mrn_col: str, acc_col: str) -> None:
	'''
	Assert 1:1 mapping between real and anon IDs in the concatenated dataframe.
	Raises RuntimeError if mapping is broken (collision or logic error).
	'''
	n_real_mrn  = concat_df[mrn_col].nunique()
	n_anon_mrn  = concat_df["anon_mrn"].nunique()
	n_real_acc  = concat_df[acc_col].nunique()
	n_anon_acc  = concat_df["anon_accession"].nunique()

	if n_real_mrn != n_anon_mrn:
		raise RuntimeError(
			f"anon_mrn mapping is not 1:1: {n_real_mrn} real MRNs → {n_anon_mrn} anon_mrns. "
			"Indicates a UUID collision or patient-map error."
		)
	if n_real_acc != n_anon_acc:
		raise RuntimeError(
			f"anon_accession mapping is not 1:1: {n_real_acc} real accessions → "
			f"{n_anon_acc} anon_accessions. Indicates a UUID collision."
		)
	print("Sanity check passed: 1:1 mapping confirmed for MRNs and accessions.")


###### Scan duplicate check ######

def check_scan_duplicates(new_df: pd.DataFrame, prior_df: pd.DataFrame, acc_col: str) -> int:
	'''
	Warn if any scan accession in new_df already exists in prior_df.
	This indicates a true duplicate scan delivery and needs manual review.
	Processing continues — returns the count of duplicates found.
	'''
	mask = new_df[acc_col].isin(prior_df[acc_col])
	n = int(mask.sum())
	if n > 0:
		dup_ids = new_df.loc[mask, acc_col].tolist()
		preview = dup_ids[:10]
		suffix = f" ... (+{n - 10} more)" if n > 10 else ""
		warnings.warn(
			f"{n} scan identifier(s) in the new batch already exist in prior data "
			f"and require manual review:\n  {preview}{suffix}",
			UserWarning,
			stacklevel=2,
		)
	return n


###### Anonymization assignment ######

def assign_anon_mrns(new_df: pd.DataFrame, prior_df: pd.DataFrame | None, mrn_col: str, mrn_len: int) -> pd.DataFrame:
	'''
	Assign anon_mrn to every row in new_df.
	- Patients already in prior_df reuse their existing anon_mrn.
	- New patients receive freshly generated UUIDs, collision-checked against
	  all anon_mrns already in existence.
	'''
	existing_anon_mrns: set = (
		set(prior_df["anon_mrn"].dropna()) if prior_df is not None else set()
	)
	prior_mrn_map: dict = (
		prior_df.drop_duplicates(mrn_col)
				.set_index(mrn_col)["anon_mrn"]
				.to_dict()
		if prior_df is not None else {}
	)

	patient_map: dict = {}
	for mrn in new_df[mrn_col].unique():
		if mrn in prior_mrn_map:
			# Returning patient — preserve their original anon_mrn
			anon = prior_mrn_map[mrn]
			existing_anon_mrns.add(anon)   # protect it from being re-issued
			patient_map[mrn] = anon
		else:
			patient_map[mrn] = generate_unique_id(existing_anon_mrns, mrn_len)

	new_df = new_df.copy()
	new_df["anon_mrn"] = new_df[mrn_col].map(patient_map)
	return new_df


def assign_anon_accessions(new_df: pd.DataFrame, prior_df: pd.DataFrame | None, acc_col: str, acc_len: int) -> pd.DataFrame:
	'''
	Assign a unique anon_accession to each scan row in new_df.
	- Scans already present in prior_df (true duplicates) reuse their existing
	  anon_accession so the 1:1 mapping is preserved. The caller is responsible
	  for having already warned about these via check_scan_duplicates().
	- New scans receive freshly generated UUIDs, collision-checked against all
	  prior anon_accessions.
	'''
	existing_anon_accs: set = (
		set(prior_df["anon_accession"].dropna()) if prior_df is not None else set()
	)
	prior_acc_map: dict = (
		prior_df.drop_duplicates(acc_col)
				.set_index(acc_col)["anon_accession"]
				.to_dict()
		if prior_df is not None else {}
	)

	anon_accs = []
	for acc in new_df[acc_col]:
		if acc in prior_acc_map:
			# Duplicate scan — preserve its original anon_accession
			existing_anon_accs.add(prior_acc_map[acc])
			anon_accs.append(prior_acc_map[acc])
		else:
			anon_accs.append(generate_unique_id(existing_anon_accs, acc_len))

	new_df = new_df.copy()
	new_df["anon_accession"] = anon_accs
	return new_df


###### Orchestration ######

def run_crosswalk(input_path: Path, prior_path: Path | None, mrn_col: str, acc_col: str, batch_label: str, output_dir: Path, mrn_len: int = 10, acc_len: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
	'''
	Full crosswalk pipeline. Returns (new_batch_df, concat_df).
	All anon columns are present on both returned dataframes.
	'''

	if not (1 <= mrn_len <= 32):
		raise ValueError(f"mrn_len must be between 1 and 32, got {mrn_len}")
	if not (1 <= acc_len <= 32):
		raise ValueError(f"acc_len must be between 1 and 32, got {acc_len}")

	### Load & validate new batch ###
	new_df = pd.read_csv(input_path)
	_validate_columns(new_df, [mrn_col, acc_col], f"new batch ({input_path.name})")
	for col in [mrn_col, acc_col]:
		if new_df[col].isna().any():
			raise ValueError(f"Column '{col}' contains NaN values in new batch — all rows must have valid IDs")

	### Load & validate prior data ###
	prior_df: pd.DataFrame | None = None
	if prior_path is not None:
		prior_df = pd.read_csv(prior_path)
		_validate_columns(
			prior_df,
			[mrn_col, acc_col, "anon_mrn", "anon_accession"],
			f"prior data ({prior_path.name})",
		)

	### Coerce ID columns to str to prevent dtype drift across batches ###
	new_df[mrn_col] = new_df[mrn_col].astype(str)
	new_df[acc_col] = new_df[acc_col].astype(str)
	if prior_df is not None:
		prior_df[mrn_col] = prior_df[mrn_col].astype(str)
		prior_df[acc_col] = prior_df[acc_col].astype(str)
		mrn_groups = prior_df.groupby(mrn_col)["anon_mrn"].nunique()
		if (mrn_groups > 1).any():
			bad_mrns = mrn_groups[mrn_groups > 1].index.tolist()
			raise RuntimeError(
				f"Prior data has conflicting anon_mrn for MRN(s): {bad_mrns}. "
				"The prior file may be corrupted or merged from mismatched sources."
			)

	###### Diagnostics ######
	print(
		f"New batch '{batch_label}': "
		f"{new_df[mrn_col].nunique()} unique patients, {len(new_df)} scans"
	)
	if prior_df is not None:
		n_overlap_pt = new_df.drop_duplicates(mrn_col)[mrn_col].isin(prior_df[mrn_col]).sum()
		print(f"Patients in new batch also seen in prior data: {n_overlap_pt}")
		n_dup_scans = check_scan_duplicates(new_df, prior_df, acc_col)
		if n_dup_scans == 0:
			print("No duplicate scan identifiers found across batches.")

	###### Assign anonymized IDs ######
	new_df = assign_anon_mrns(new_df, prior_df, mrn_col, mrn_len)
	new_df = assign_anon_accessions(new_df, prior_df, acc_col, acc_len)
	new_df["batch"] = batch_label

	###### Concatenate with prior ######
	if prior_df is not None:
		if "batch" not in prior_df.columns:
			prior_df = prior_df.copy()
			prior_df["batch"] = "prior"
		concat_df = pd.concat([prior_df, new_df], ignore_index=True)
	else:
		concat_df = new_df.copy()

	###### Sanity check ######
	_sanity_check(concat_df, mrn_col, acc_col)

	###### Write outputs ######
	output_dir.mkdir(parents=True, exist_ok=True)
	anon_cols = [mrn_col, acc_col, "anon_mrn", "anon_accession"]

	batch_anon_path  = output_dir / f"{batch_label}_batch_anon.csv"
	concat_full_path = output_dir / f"{batch_label}_concat_full.csv"
	concat_anon_path = output_dir / f"{batch_label}_concat_anon.csv"

	new_df[anon_cols].to_csv(batch_anon_path, index=False)
	concat_df.to_csv(concat_full_path, index=False)
	concat_df[anon_cols].to_csv(concat_anon_path, index=False)

	print(f"Wrote: {batch_anon_path}")
	print(f"Wrote: {concat_full_path}")
	print(f"Wrote: {concat_anon_path}")

	###### Summary ######
	print(
		f"Cumulative totals — "
		f"unique MRNs: {concat_df[mrn_col].nunique()}, "
		f"unique anon_mrns: {concat_df['anon_mrn'].nunique()}, "
		f"unique accessions: {concat_df[acc_col].nunique()}, "
		f"unique anon_accessions: {concat_df['anon_accession'].nunique()}"
	)

	return new_df, concat_df



def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Crosswalk generator v2.0",
		epilog="Version 2.0; Created by Rohan Shad, MD"
	)

	p.add_argument("--input",       required=True,          help="New batch CSV file path")
	p.add_argument("--prior",       default=None,           help="Prior cumulative full CSV ({batch_label}_concat_full.csv from a previous run)")
	p.add_argument("--mrn-col",     default="Patient ID",   help="Column name for patient MRN (default: 'Patient ID')")
	p.add_argument("--acc-col",     default="Study ID",     help="Column name for scan accession/UID (default: 'Study ID')")
	p.add_argument("--batch-label", default="batch",        help="Label for this batch — used in output filenames and the 'batch' column (default: 'batch')")
	p.add_argument("--output-dir",  default=".",            help="Directory for output files (default: current directory)")
	p.add_argument("--mrn-len",     type=int, default=10,   help="Hex character length for anon_mrn (default: 10)")
	p.add_argument("--acc-len",     type=int, default=12,   help="Hex character length for anon_accession (default: 12)")
	return p


def main() -> None:
	args = build_parser().parse_args()
	print('------------------------------------')
	print(f'Input PHI csv:          {args.input}')
	print(f'Existing crosswalk csv: {args.prior}')
	print('------------------------------------')
	run_crosswalk(
		input_path=Path(args.input),
		prior_path=Path(args.prior) if args.prior else None,
		mrn_col=args.mrn_col,
		acc_col=args.acc_col,
		batch_label=args.batch_label,
		output_dir=Path(args.output_dir),
		mrn_len=args.mrn_len,
		acc_len=args.acc_len,
	)


if __name__ == "__main__":
	main()



