"""Tests for portable configuration and BIDS subject discovery."""

import importlib
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def test_environment_configuration(monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ("raw", "processed", "reports", "figures")}
    monkeypatch.setenv("MTBI_RAW_DATA_DIR", str(paths["raw"]))
    monkeypatch.setenv("MTBI_PROCESSED_DATA_DIR", str(paths["processed"]))
    monkeypatch.setenv("MTBI_REPORTS_DIR", str(paths["reports"]))
    monkeypatch.setenv("MTBI_FIGURES_DIR", str(paths["figures"]))
    monkeypatch.setenv("MTBI_N_JOBS", "2")
    monkeypatch.setenv("MTBI_MPL_BACKEND", "Agg")

    import config_common
    config_common = importlib.reload(config_common)

    assert config_common.raw_data_dir == str(paths["raw"])
    assert config_common.processed_data_dir == str(paths["processed"])
    assert config_common.reports_dir == str(paths["reports"])
    assert config_common.figures_dir == str(paths["figures"])
    assert config_common.n_jobs == 2


def test_discover_subjects_handles_missing_and_bids_directories(tmp_path):
    import config_eeg

    assert config_eeg.discover_subjects(tmp_path / "missing") == []
    (tmp_path / "sub-02P").mkdir()
    (tmp_path / "sub-01C").mkdir()
    (tmp_path / "participants.tsv").write_text("participant_id\n")
    (tmp_path / "notes").mkdir()

    assert config_eeg.discover_subjects(tmp_path) == ["01C", "02P"]
