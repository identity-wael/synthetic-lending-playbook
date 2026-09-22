"""Decision calibration and domain shift audit for Synthetic Geometry Classifiers. Apache-2.0."""
import os, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_model_directory():
    """Locate model directory via environment, input mount, local repo, or kagglehub."""
    candidates = [
        os.environ.get('MODEL_ROOT'),
        '/kaggle/input/synthetic-geometry-classifiers/other/comparison-bundle/1',
        '/kaggle/input/synthetic-geometry-classifiers',
        '/kaggle/input',
        'geometry-demo/model_exports',
        'model_exports'
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            p = sorted(Path(c).rglob('model.json'))
            if p:
                return Path(c)
    # Fallback to kagglehub if available
    try:
        import kagglehub
        path = kagglehub.model_download('waelelghazzawi/synthetic-geometry-classifiers/other/comparison-bundle/1')
        if Path(path).is_dir():
            return Path(path)
    except Exception as e:
        print(f"kagglehub model download fallback skipped: {e}")
    raise FileNotFoundError("Could not locate synthetic-geometry-classifiers model exports")

def predict(x, m):
    """Predict calibrated probability of class 1 for 2D synthetic coordinates."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Expected N-by-2 coordinates')
    v = m['variant']
    if v == 'linear':
        f = x
    elif v == 'quadratic':
        f = np.column_stack([x, x[:, 0]**2, x[:, 0] * x[:, 1], x[:, 1]**2])
    elif v == 'radial':
        f = (x * x).sum(axis=1).reshape(-1, 1)
    else:
        raise ValueError(f'Unknown variant: {v}')
    z = f @ np.asarray(m['coefficients']) + m['intercept']
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def compute_calibration(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    bin_records = []
    
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            diff = abs(acc - conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
            bin_records.append({
                'bin': b,
                'count': count,
                'bin_center': float((bin_edges[b] + bin_edges[b+1]) / 2.0),
                'mean_confidence': round(conf, 4),
                'empirical_frequency': round(acc, 4)
            })
            
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    return {
        'brier_score': float(brier_score),
        'ece': float(ece),
        'mce': float(mce),
        'bins': bin_records
    }

def run_calibration_audit():
    model_dir = get_model_directory()
    paths = sorted(model_dir.rglob('model.json'))
    assert paths, f"No model.json files found in {model_dir}"

    models = {}
    for path in paths:
        m = json.loads(path.read_text())
        v = json.loads(path.with_name('test_vectors.json').read_text())
        np.testing.assert_allclose(predict(v['inputs'], m), v['probability_class_1'], atol=1e-12, rtol=1e-12)
        if m['variant'] not in models:
            models[m['variant']] = m

    assert set(models) == {'linear', 'quadratic', 'radial'}

    # Reconstruct fixed held-out split (seed 1203, N=2000)
    rng = np.random.default_rng(1203)
    x_test = rng.uniform(-1, 1, (2000, 2))
    clean = ((x_test * x_test).sum(axis=1) < 0.5).astype(int)
    y_test = np.where(rng.random(2000) < 0.05, 1 - clean, clean)

    # 1. Calibration on held-out split
    calibration_summary = []
    cal_details = {}
    for name in ['linear', 'quadratic', 'radial']:
        m = models[name]
        p = predict(x_test, m)
        cal = compute_calibration(y_test, p)
        cal_details[name] = cal
        row = {
            'variant': name,
            'brier_score': round(cal['brier_score'], 5),
            'ece': round(cal['ece'], 5),
            'mce': round(cal['mce'], 5),
            'brier_reduction_pct': round((1.0 - cal['brier_score'] / cal_details['linear']['brier_score']) * 100.0, 2)
        }
        calibration_summary.append(row)

    # 2. Out-of-Distribution (OOD) Domain Shift Evaluation
    x_ood_raw = rng.uniform(-2, 2, (5000, 2))
    is_outside_box = (np.abs(x_ood_raw[:, 0]) > 1.0) | (np.abs(x_ood_raw[:, 1]) > 1.0)
    x_ood = x_ood_raw[is_outside_box][:2000]
    ood_summary = []
    for name in ['linear', 'quadratic', 'radial']:
        m = models[name]
        p_ood = predict(x_ood, m)
        ood_summary.append({
            'variant': name,
            'ood_sample_size': len(x_ood),
            'ood_mean_prob': round(float(np.mean(p_ood)), 5),
            'ood_max_prob': round(float(np.max(p_ood)), 5),
            'ood_false_positive_rate': round(float(np.mean(p_ood >= m['threshold'])), 5)
        })

    # 3. Export CSV and JSON artifacts
    with open('calibration_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(calibration_summary[0]))
        w.writeheader()
        w.writerows(calibration_summary)

    with open('ood_shift_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ood_summary[0]))
        w.writeheader()
        w.writerows(ood_summary)

    with open('calibration_metrics.json', 'w') as f:
        json.dump({
            'calibration': cal_details,
            'summary': calibration_summary,
            'ood_shift': ood_summary
        }, f, indent=2)

    # 4. Generate Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Reliability Curves
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.6)
    colors = {'linear': '#1f77b4', 'quadratic': '#ff7f0e', 'radial': '#2ca02c'}
    markers = {'linear': 'o', 'quadratic': 's', 'radial': '^'}
    for name in ['linear', 'quadratic', 'radial']:
        bins = cal_details[name]['bins']
        confs = [b['mean_confidence'] for b in bins]
        accs = [b['empirical_frequency'] for b in bins]
        ax1.plot(confs, accs, marker=markers[name], color=colors[name],
                 label=f"{name} (Brier: {cal_details[name]['brier_score']:.3f}, ECE: {cal_details[name]['ece']:.3f})")
    ax1.set_xlabel('Predicted Probability (Confidence)')
    ax1.set_ylabel('Empirical Class 1 Frequency')
    ax1.set_title('Reliability Diagrams (Calibration)')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel B: Radial Margin Profile
    radii = np.linspace(0.05, 1.0, 50)
    # Coordinates along diagonal x1 = x2 = r / sqrt(2)
    coords = np.column_stack([radii / np.sqrt(2), radii / np.sqrt(2)])
    boundary_r = np.sqrt(0.5)
    for name in ['linear', 'quadratic', 'radial']:
        p_profile = predict(coords, models[name])
        ax2.plot(radii, p_profile, label=name, color=colors[name], linewidth=2)

    # True generative probability step function (with 5% flip rate: 0.95 inside, 0.05 outside)
    true_probs = np.where(radii < boundary_r, 0.95, 0.05)
    ax2.plot(radii, true_probs, 'k--', label='Generative Ground Truth (5% flip)', alpha=0.7)
    ax2.axvline(boundary_r, color='gray', linestyle=':', label=f'True Boundary (r={boundary_r:.3f})')
    ax2.set_xlabel('Radial Distance from Origin (r)')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Confidence vs. Radial Boundary Distance')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(json.dumps(calibration_summary, indent=2))
    print(json.dumps(ood_summary, indent=2))
    print("PASS: Calibration and domain shift audit completed successfully.")

if __name__ == '__main__':
    run_calibration_audit()
