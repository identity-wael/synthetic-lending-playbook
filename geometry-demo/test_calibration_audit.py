"""Unit and regression tests for calibration_audit.py. Apache-2.0."""
import os, json
from pathlib import Path
import numpy as np
import pytest

from calibration_audit import get_model_directory, predict, compute_calibration, run_calibration_audit

def test_model_loading_and_test_vectors():
    model_dir = get_model_directory()
    paths = sorted(model_dir.rglob('model.json'))
    assert len(paths) >= 3, f"Expected at least 3 models, found {len(paths)}"
    
    models = {}
    for path in paths:
        m = json.loads(path.read_text())
        v = json.loads(path.with_name('test_vectors.json').read_text())
        p = predict(v['inputs'], m)
        np.testing.assert_allclose(p, v['probability_class_1'], atol=1e-12, rtol=1e-12)
        models[m['variant']] = m
        
    assert 'linear' in models
    assert 'quadratic' in models
    assert 'radial' in models

def test_calibration_computation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    cal = compute_calibration(y_true, y_prob, n_bins=5)
    assert 'brier_score' in cal
    assert 'ece' in cal
    assert 'mce' in cal
    assert cal['brier_score'] < 0.05
    assert cal['ece'] < 0.2

def test_invalid_coordinates():
    model_dir = get_model_directory()
    path = next(model_dir.rglob('model.json'))
    m = json.loads(path.read_text())
    
    # 1D array instead of 2D
    with pytest.raises(ValueError, match="Expected N-by-2"):
        predict([0.5, 0.5], m)
        
    # 3D coordinates instead of 2D
    with pytest.raises(ValueError, match="Expected N-by-2"):
        predict([[0.5, 0.5, 0.5]], m)

def test_full_audit_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_calibration_audit()
    
    assert (tmp_path / 'calibration_summary.csv').is_file()
    assert (tmp_path / 'ood_shift_summary.csv').is_file()
    assert (tmp_path / 'calibration_metrics.json').is_file()
    assert (tmp_path / 'calibration_analysis.png').is_file()
    
    metrics = json.loads((tmp_path / 'calibration_metrics.json').read_text())
    assert metrics['calibration']['radial']['brier_score'] < 0.070
    assert metrics['calibration']['quadratic']['brier_score'] < 0.070
    assert metrics['calibration']['linear']['brier_score'] > 0.240
    
    # OOD metrics
    ood = {r['variant']: r for r in metrics['ood_shift']}
    assert ood['radial']['ood_mean_prob'] < 0.001
    assert ood['linear']['ood_mean_prob'] > 0.350
