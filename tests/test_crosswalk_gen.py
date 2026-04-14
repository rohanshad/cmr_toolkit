"""
test_crosswalk_gen.py — pytest suite for crosswalk_gen.py

Fake data layout
----------------
Batch 1 — 4 patients, 6 scans:
  P1: S1, S2        (2 scans)
  P2: S3
  P3: S4, S5        (2 scans)
  P4: S6

Batch 2 — 4 patients, 5 scans (P3, P4 return; P5, P6 new):
  P3: S7            (returning patient, new scan)
  P4: S8            (returning patient, new scan)
  P5: S9
  P6: S10, S11      (2 scans)

Batch 2 with deliberate collision — same as batch 2 but S1 (from batch 1) is included:
  P3: S1            <- duplicate scan ID across batches → should trigger UserWarning
  ...rest same
"""

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import crosswalk_gen as cw


# ── Fixtures ───────────────────────────────────────────────────────────────────

MRN_COL   = "Patient ID"
ACC_COL   = "Study ID"
EXTRA_COLS = ["Report", "Age"]   # non-ID payload columns; update here if the fixture schema changes


@pytest.fixture
def batch1_df() -> pd.DataFrame:
    report_col, age_col = EXTRA_COLS
    return pd.DataFrame({
        MRN_COL:    ["P1", "P1", "P2", "P3", "P3", "P4"],
        ACC_COL:    ["S1", "S2", "S3", "S4", "S5", "S6"],
        report_col: ["r1", "r2", "r3", "r4", "r5", "r6"],
        age_col:    [55,   55,   62,   47,   47,   70],
    })


@pytest.fixture
def batch2_df() -> pd.DataFrame:
    report_col, age_col = EXTRA_COLS
    return pd.DataFrame({
        MRN_COL:    ["P3", "P4", "P5", "P6", "P6"],
        ACC_COL:    ["S7", "S8", "S9", "S10", "S11"],
        report_col: ["r7", "r8", "r9", "r10", "r11"],
        age_col:    [48,   71,   33,   29,    29],
    })


@pytest.fixture
def batch2_with_dup_scan_df() -> pd.DataFrame:
    """Batch 2 where one scan ID (S1) duplicates a batch 1 scan."""
    report_col, age_col = EXTRA_COLS
    return pd.DataFrame({
        MRN_COL:    ["P3", "P4", "P5", "P6", "P6"],
        ACC_COL:    ["S1", "S8", "S9", "S10", "S11"],   # S1 is a dup
        report_col: ["r7", "r8", "r9", "r10", "r11"],
        age_col:    [48,   71,   33,   29,    29],
    })


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ── Helper: run batch 1, return paths ─────────────────────────────────────────

def run_batch1(tmp_path, batch1_df):
    b1_csv = write_csv(batch1_df, tmp_path / "batch1.csv")
    new_df, concat_df = cw.run_crosswalk(
        input_path=b1_csv,
        prior_path=None,
        mrn_col=MRN_COL,
        acc_col=ACC_COL,
        batch_label="batch_1",
        output_dir=tmp_path / "out",
    )
    concat_full_path = tmp_path / "out" / "batch_1_concat_full.csv"
    return new_df, concat_df, concat_full_path


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFirstBatch:

    def test_three_output_files_created(self, tmp_path, batch1_df):
        run_batch1(tmp_path, batch1_df)
        out = tmp_path / "out"
        assert (out / "batch_1_batch_anon.csv").exists()
        assert (out / "batch_1_concat_full.csv").exists()
        assert (out / "batch_1_concat_anon.csv").exists()

    def test_batch_anon_has_correct_columns(self, tmp_path, batch1_df):
        run_batch1(tmp_path, batch1_df)
        df = pd.read_csv(tmp_path / "out" / "batch_1_batch_anon.csv")
        assert set(df.columns) == {MRN_COL, ACC_COL, "anon_mrn", "anon_accession"}

    def test_batch_anon_row_count(self, tmp_path, batch1_df):
        run_batch1(tmp_path, batch1_df)
        df = pd.read_csv(tmp_path / "out" / "batch_1_batch_anon.csv")
        assert len(df) == len(batch1_df)

    def test_anon_mrn_patient_level(self, tmp_path, batch1_df):
        """Same patient → same anon_mrn within a single batch."""
        new_df, _, _ = run_batch1(tmp_path, batch1_df)
        p1_anons = new_df.loc[new_df[MRN_COL] == "P1", "anon_mrn"].unique()
        assert len(p1_anons) == 1

    def test_anon_accession_unique_per_scan(self, tmp_path, batch1_df):
        new_df, _, _ = run_batch1(tmp_path, batch1_df)
        assert new_df["anon_accession"].nunique() == len(new_df)

    def test_no_prior_concat_equals_batch(self, tmp_path, batch1_df):
        """Without prior data the concat output == the batch output."""
        _, concat_df, _ = run_batch1(tmp_path, batch1_df)
        assert len(concat_df) == len(batch1_df)

    def test_batch_column_stamped(self, tmp_path, batch1_df):
        """new_df must have 'batch' column set to the batch_label."""
        new_df, _, _ = run_batch1(tmp_path, batch1_df)
        assert (new_df["batch"] == "batch_1").all()


class TestReturningPatients:

    def test_returning_patient_reuses_anon_mrn(self, tmp_path, batch1_df, batch2_df):
        """P3 and P4 appear in both batches — they must keep their batch 1 anon_mrn."""
        _, b1_concat, b1_full_path = run_batch1(tmp_path, batch1_df)

        b1_mrn_map = (
            b1_concat.drop_duplicates(MRN_COL)
                      .set_index(MRN_COL)["anon_mrn"]
                      .to_dict()
        )

        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        new_df, _ = cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )

        for patient in ["P3", "P4"]:
            b2_anon = new_df.loc[new_df[MRN_COL] == patient, "anon_mrn"].iloc[0]
            assert b2_anon == b1_mrn_map[patient], (
                f"{patient} got a different anon_mrn in batch 2"
            )

    def test_new_patient_gets_fresh_anon_mrn(self, tmp_path, batch1_df, batch2_df):
        """P5 and P6 are new — their anon_mrns must not appear in batch 1."""
        _, b1_concat, b1_full_path = run_batch1(tmp_path, batch1_df)
        b1_anon_mrns = set(b1_concat["anon_mrn"].unique())

        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        new_df, _ = cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )

        for patient in ["P5", "P6"]:
            b2_anon = new_df.loc[new_df[MRN_COL] == patient, "anon_mrn"].iloc[0]
            assert b2_anon not in b1_anon_mrns, (
                f"{patient} (new patient) was assigned an anon_mrn that already existed"
            )

    def test_concat_anon_mrn_1to1_mapping(self, tmp_path, batch1_df, batch2_df):
        """After two batches, nunique(real_mrn) == nunique(anon_mrn) in concat."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        _, concat_df = cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )
        assert concat_df[MRN_COL].nunique() == concat_df["anon_mrn"].nunique()
        assert concat_df[ACC_COL].nunique() == concat_df["anon_accession"].nunique()

    def test_concat_row_count(self, tmp_path, batch1_df, batch2_df):
        """Total rows in concat == batch1 + batch2."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        _, concat_df = cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )
        assert len(concat_df) == len(batch1_df) + len(batch2_df)

    def test_int_mrn_matches_str_mrn_across_batches(self, tmp_path):
        """Numeric MRNs in prior CSV must match str MRNs in new batch — guards dtype coercion."""
        # Batch 1: all-numeric MRNs → pandas reads the saved CSV back as int64
        b1 = pd.DataFrame({MRN_COL: [101, 102], ACC_COL: ["S1", "S2"]})
        b1_csv = write_csv(b1, tmp_path / "b1.csv")
        _, b1_concat = cw.run_crosswalk(
            input_path=b1_csv, prior_path=None,
            mrn_col=MRN_COL, acc_col=ACC_COL,
            batch_label="batch_1", output_dir=tmp_path / "out1",
        )
        b1_full_path = tmp_path / "out1" / "batch_1_concat_full.csv"
        b1_anon_for_101 = b1_concat.loc[b1_concat[MRN_COL].astype(str) == "101", "anon_mrn"].iloc[0]

        # Batch 2: alphanumeric new patient forces the MRN column to object dtype,
        # so returning patient 101 arrives as str "101" instead of int 101.
        b2 = pd.DataFrame({MRN_COL: ["101", "ABC103"], ACC_COL: ["S3", "S4"]})
        b2_csv = write_csv(b2, tmp_path / "b2.csv")
        new_df2, _ = cw.run_crosswalk(
            input_path=b2_csv, prior_path=b1_full_path,
            mrn_col=MRN_COL, acc_col=ACC_COL,
            batch_label="batch_2", output_dir=tmp_path / "out2",
        )
        b2_anon_for_101 = new_df2.loc[new_df2[MRN_COL] == "101", "anon_mrn"].iloc[0]
        assert b2_anon_for_101 == b1_anon_for_101, \
            "Patient 101 must keep the same anon_mrn even when MRN dtype drifts from int to str"


class TestUUIDCollisionRegeneration:

    def test_mrn_collision_regenerated(self, tmp_path, batch1_df):
        """
        Monkeypatch uuid.uuid4 to return a colliding hex on the first call,
        then a valid one. Verifies generate_unique_id loops and resolves.
        """
        collision_hex = "aaaa" * 10          # guaranteed to collide if pre-seeded
        valid_hex     = "bbbb" * 10

        call_count = {"n": 0}

        def fake_uuid4():
            mock = MagicMock()
            if call_count["n"] == 0:
                mock.hex = collision_hex
            else:
                mock.hex = valid_hex
            call_count["n"] += 1
            return mock

        existing = {collision_hex[:10]}       # pre-seed the collision
        with patch("crosswalk_gen.uuid.uuid4", side_effect=fake_uuid4):
            result = cw.generate_unique_id(existing, length=10)

        assert result == valid_hex[:10]
        assert call_count["n"] == 2           # called twice: collision then success
        assert result in existing

    def test_acc_collision_regenerated(self, tmp_path):
        """Same pattern for accession-length IDs."""
        collision_hex = "cccc" * 10
        valid_hex     = "dddd" * 10
        call_count = {"n": 0}

        def fake_uuid4():
            mock = MagicMock()
            mock.hex = collision_hex if call_count["n"] == 0 else valid_hex
            call_count["n"] += 1
            return mock

        existing = {collision_hex[:12]}
        with patch("crosswalk_gen.uuid.uuid4", side_effect=fake_uuid4):
            result = cw.generate_unique_id(existing, length=12)

        assert result == valid_hex[:12]
        assert call_count["n"] == 2

    def test_exhausted_attempts_raises(self):
        """If every attempt collides, RuntimeError must be raised."""
        always_same = "aaaa" * 10

        def fake_uuid4():
            mock = MagicMock()
            mock.hex = always_same
            return mock

        existing = {always_same[:10]}
        with patch("crosswalk_gen.uuid.uuid4", side_effect=fake_uuid4):
            with pytest.raises(RuntimeError, match="Could not generate a unique ID"):
                cw.generate_unique_id(existing, length=10, max_attempts=5)


class TestDuplicateScanWarning:

    def test_duplicate_scan_raises_user_warning(
        self, tmp_path, batch1_df, batch2_with_dup_scan_df
    ):
        """Scan ID S1 appears in both batches → UserWarning with S1 in message."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_with_dup_scan_df, tmp_path / "batch2_dup.csv")

        with pytest.warns(UserWarning, match="S1"):
            cw.run_crosswalk(
                input_path=b2_csv,
                prior_path=b1_full_path,
                mrn_col=MRN_COL,
                acc_col=ACC_COL,
                batch_label="batch_2",
                output_dir=tmp_path / "out2",
            )

    def test_duplicate_scan_does_not_halt_processing(
        self, tmp_path, batch1_df, batch2_with_dup_scan_df
    ):
        """Duplicate scan warning must not prevent output files from being created."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_with_dup_scan_df, tmp_path / "batch2_dup.csv")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cw.run_crosswalk(
                input_path=b2_csv,
                prior_path=b1_full_path,
                mrn_col=MRN_COL,
                acc_col=ACC_COL,
                batch_label="batch_2",
                output_dir=tmp_path / "out2",
            )

        assert (tmp_path / "out2" / "batch_2_concat_full.csv").exists()

    def test_no_warning_when_no_duplicates(self, tmp_path, batch1_df, batch2_df):
        """Clean batch 2 must not emit any UserWarning."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cw.run_crosswalk(
                input_path=b2_csv,
                prior_path=b1_full_path,
                mrn_col=MRN_COL,
                acc_col=ACC_COL,
                batch_label="batch_2",
                output_dir=tmp_path / "out2",
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


class TestValidationErrors:

    def test_wrong_mrn_col_raises_value_error(self, tmp_path, batch1_df):
        b1_csv = write_csv(batch1_df, tmp_path / "batch1.csv")
        with pytest.raises(ValueError, match="Missing columns"):
            cw.run_crosswalk(
                input_path=b1_csv,
                prior_path=None,
                mrn_col="nonexistent_mrn",
                acc_col=ACC_COL,
                batch_label="batch_1",
                output_dir=tmp_path / "out",
            )

    def test_wrong_acc_col_raises_value_error(self, tmp_path, batch1_df):
        b1_csv = write_csv(batch1_df, tmp_path / "batch1.csv")
        with pytest.raises(ValueError, match="Missing columns"):
            cw.run_crosswalk(
                input_path=b1_csv,
                prior_path=None,
                mrn_col=MRN_COL,
                acc_col="nonexistent_acc",
                batch_label="batch_1",
                output_dir=tmp_path / "out",
            )

    def test_prior_missing_anon_mrn_col_raises(self, tmp_path, batch1_df, batch2_df):
        """Prior CSV that lacks anon_mrn (e.g. wrong file passed) → ValueError."""
        # Write a prior CSV that has mrn/acc cols but no anon columns
        bad_prior = write_csv(batch1_df, tmp_path / "bad_prior.csv")
        b2_csv    = write_csv(batch2_df, tmp_path / "batch2.csv")

        with pytest.raises(ValueError, match="anon_mrn"):
            cw.run_crosswalk(
                input_path=b2_csv,
                prior_path=bad_prior,
                mrn_col=MRN_COL,
                acc_col=ACC_COL,
                batch_label="batch_2",
                output_dir=tmp_path / "out2",
            )

    def test_nan_in_mrn_raises_value_error(self, tmp_path):
        """NaN in mrn_col must be rejected before anonymization begins."""
        nan_df = pd.DataFrame({
            MRN_COL: ["P1", None, "P3"],
            ACC_COL: ["S1", "S2",  "S3"],
        })
        csv = write_csv(nan_df, tmp_path / "nan_mrn.csv")
        with pytest.raises(ValueError, match=MRN_COL):
            cw.run_crosswalk(
                input_path=csv, prior_path=None,
                mrn_col=MRN_COL, acc_col=ACC_COL,
                batch_label="batch_1", output_dir=tmp_path / "out",
            )

    def test_nan_in_acc_raises_value_error(self, tmp_path):
        """NaN in acc_col must be rejected before anonymization begins."""
        nan_df = pd.DataFrame({
            MRN_COL: ["P1", "P2",  "P3"],
            ACC_COL: ["S1", None,  "S3"],
        })
        csv = write_csv(nan_df, tmp_path / "nan_acc.csv")
        with pytest.raises(ValueError, match=ACC_COL):
            cw.run_crosswalk(
                input_path=csv, prior_path=None,
                mrn_col=MRN_COL, acc_col=ACC_COL,
                batch_label="batch_1", output_dir=tmp_path / "out",
            )


class TestSanityCheck:

    def test_sanity_check_passes_silently(self, tmp_path, batch1_df):
        """Normal run must not raise from _sanity_check."""
        b1_csv = write_csv(batch1_df, tmp_path / "batch1.csv")
        # Should complete without raising
        cw.run_crosswalk(
            input_path=b1_csv,
            prior_path=None,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_1",
            output_dir=tmp_path / "out",
        )

    def test_sanity_check_catches_broken_mrn_mapping(self):
        """Inject a duplicate anon_mrn into a concat df → RuntimeError."""
        broken = pd.DataFrame({
            MRN_COL:         ["P1", "P2"],
            ACC_COL:         ["S1", "S2"],
            "anon_mrn":      ["aaa", "aaa"],   # duplicate — two patients, one anon_mrn
            "anon_accession":["x1",  "x2"],
        })
        with pytest.raises(RuntimeError, match="anon_mrn mapping is not 1:1"):
            cw._sanity_check(broken, MRN_COL, ACC_COL)

    def test_sanity_check_catches_broken_acc_mapping(self):
        """Inject a duplicate anon_accession → RuntimeError."""
        broken = pd.DataFrame({
            MRN_COL:         ["P1", "P2"],
            ACC_COL:         ["S1", "S2"],
            "anon_mrn":      ["aaa", "bbb"],
            "anon_accession":["x1",  "x1"],    # duplicate
        })
        with pytest.raises(RuntimeError, match="anon_accession mapping is not 1:1"):
            cw._sanity_check(broken, MRN_COL, ACC_COL)


class TestOutputFileContents:

    def test_batch_anon_contains_only_new_rows(self, tmp_path, batch1_df, batch2_df):
        """batch_anon output must contain exactly the new batch rows."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )
        batch_anon = pd.read_csv(tmp_path / "out2" / "batch_2_batch_anon.csv")
        assert len(batch_anon) == len(batch2_df)
        assert set(batch_anon[ACC_COL]) == set(batch2_df[ACC_COL])

    def test_concat_full_has_all_columns(self, tmp_path, batch1_df, batch2_df):
        """concat_full must carry through all original columns from both batches."""
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )
        concat_full = pd.read_csv(tmp_path / "out2" / "batch_2_concat_full.csv")
        for col in [MRN_COL, ACC_COL, "anon_mrn", "anon_accession", *EXTRA_COLS, "batch"]:
            assert col in concat_full.columns, f"Expected column '{col}' missing from concat_full"

    def test_concat_anon_has_only_four_columns(self, tmp_path, batch1_df, batch2_df):
        _, _, b1_full_path = run_batch1(tmp_path, batch1_df)
        b2_csv = write_csv(batch2_df, tmp_path / "batch2.csv")
        cw.run_crosswalk(
            input_path=b2_csv,
            prior_path=b1_full_path,
            mrn_col=MRN_COL,
            acc_col=ACC_COL,
            batch_label="batch_2",
            output_dir=tmp_path / "out2",
        )
        concat_anon = pd.read_csv(tmp_path / "out2" / "batch_2_concat_anon.csv")
        assert set(concat_anon.columns) == {MRN_COL, ACC_COL, "anon_mrn", "anon_accession"}
