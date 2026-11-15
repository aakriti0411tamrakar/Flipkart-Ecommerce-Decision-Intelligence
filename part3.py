"""Part 3: reproducible NumPy-only segmentation, linkage, and PCA."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
FEATURES = ['customer_tenure_months', 'complaint_count_90d', 'discount_order_share', 'avg_order_value_inr_hundreds', 'loyalty_score']


def kmeans(Z, k):
    n = len(Z)
    centroids = Z[np.arange(k) * (n // k)].copy()
    previous = np.full(n, -1, dtype=int)
    passes = 0
    while True:
        passes += 1
        dist = np.sum((Z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist, axis=1)
        if np.array_equal(labels, previous):
            break
        for j in range(k):
            members = Z[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
        previous = labels.copy()
    wcss = float(np.sum((Z - centroids[labels]) ** 2))
    return labels, centroids, wcss, passes


def linkage_distance(a, b, dist, mode):
    values = [dist[i, j] for i in a for j in b]
    return {'single': min, 'complete': max, 'average': lambda x: sum(x) / len(x)}[mode](values)


def hierarchical(mode, dist):
    clusters = [(i,) for i in range(6)]
    rows = []
    for step in range(1, 6):
        candidates = [(linkage_distance(a, b, dist, mode), i, j)
                      for i, a in enumerate(clusters) for j, b in enumerate(clusters) if i < j]
        height, i, j = min(candidates)
        a, b = clusters[i], clusters[j]
        if a[0] > b[0]: a, b = b, a
        name = lambda members: '+'.join(f'E{v+1}' for v in members)
        rows.append((mode, step, name(a), name(b), float(height)))
        clusters = [c for t, c in enumerate(clusters) if t not in (i, j)] + [tuple(sorted(a + b))]
        clusters.sort()
    gaps = [rows[i+1][4] - rows[i][4] for i in range(4)]
    merges_below_cut = int(np.argmax(gaps)) + 1
    return rows, 6 - merges_below_cut


def write_csv(path, header, rows):
    with (OUT / path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f'{float(v):.4f}' if isinstance(v, (float, np.floating)) else v for v in row])


def main():
    OUT.mkdir(exist_ok=True)
    with (ROOT / 'data/records.csv').open(newline='') as f:
        data = list(csv.DictReader(f))
    X = np.array([[float(row[c]) for c in FEATURES] for row in data])
    outcome = np.array([int(row['churned']) for row in data])
    value = np.array([float(row['order_value_inr_hundreds']) for row in data])
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    results = {}
    solutions = {}
    for k in range(2, 7):
        labels, centers, wcss, passes = kmeans(Z, k)
        solutions[k] = (labels, centers, wcss, passes)
        results[f'wcss_k{k}'] = wcss
    assert abs(results['wcss_k2'] - 2262.6753) < 0.0001
    results['elbow_k'] = next((k for k in (3, 4, 5) if
        results[f'wcss_k{k}'] - results[f'wcss_k{k+1}'] <
        0.5 * (results[f'wcss_k{k-1}'] - results[f'wcss_k{k}'])), 0)
    labels = solutions[3][0]
    ordering = sorted(range(3), key=lambda c: float(X[labels == c, 2].mean()))
    segments = {}
    seg_rows = []
    for name, cluster in zip('ABC', ordering):
        mask = labels == cluster
        segments[name] = mask
        n = int(mask.sum())
        results[f'size_segment_{name}'] = n
        seg_rows.append((name, n, *X[mask].mean(axis=0), float(outcome[mask].mean()), float(value[mask].mean())))
    write_csv('part3_segments.csv', ['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'], seg_rows)
    D = np.array([[0,7.9,7.4,2.9,2.9,6.1], [7.9,0,1.4,5.9,5.3,2.0],
                  [7.4,1.4,0,5.8,4.6,1.3], [2.9,5.9,5.8,0,2.8,4.5],
                  [2.9,5.3,4.6,2.8,0,3.4], [6.1,2.0,1.3,4.5,3.4,0]], dtype=float)
    hier_rows = []
    for mode in ('single', 'complete', 'average'):
        rows, count = hierarchical(mode, D)
        hier_rows.extend(rows)
        results[f'{mode}_clusters_at_cut'] = count
    write_csv('part3_hierarchical.csv', ['linkage','merge_no','cluster_a','cluster_b','height'], hier_rows)
    centered = Z - Z.mean(axis=0)
    covariance = centered.T @ centered / (len(Z) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    for col in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, col]
        if vec[np.argmax(np.abs(vec))] < 0:
            eigenvectors[:, col] *= -1
    projection = centered @ eigenvectors[:, :2]
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    results['eigenvalue_sum'] = total
    for i in range(5):
        results[f'eigenvalue_{i+1}'] = float(eigenvalues[i])
        results[f'explained_ratio_{i+1}'] = float(ratios[i])
        results[f'cumulative_ratio_{i+1}'] = float(cumulative[i])
        results[f'pc1_loading_X{i+1}'] = float(eigenvectors[i, 0])
        results[f'pc2_loading_X{i+1}'] = float(eigenvectors[i, 1])
    results['components_for_80pct'] = int(np.searchsorted(cumulative, 0.8) + 1)
    for row in range(3):
        for component in range(2):
            results[f'row{row+1}_pc{component+1}'] = float(projection[row, component])
    for name in 'ABC':
        results[f'mean_pc1_segment_{name}'] = float(projection[segments[name], 0].mean())
    assert abs(total - 5.0) < 1e-10
    assert sum(results[f'size_segment_{name}'] for name in 'ABC') == 800
    with (OUT / 'part3_results.txt').open('w') as f:
        for key, v in results.items():
            f.write(f'{key} = {v:.4f}\n' if isinstance(v, (float, np.floating)) else f'{key} = {v}\n')
    print('K-means passes:', {k: solutions[k][3] for k in solutions})
    print('WCSS:', {k: round(solutions[k][2], 4) for k in solutions})
    print('Elbow:', results['elbow_k'])
    print('Segments:', [(r[0], r[1], round(r[7], 4)) for r in seg_rows])
    print('Cut clusters:', {m: results[f'{m}_clusters_at_cut'] for m in ('single','complete','average')})
    print('PCA variance:', np.round(ratios, 4), 'components80:', results['components_for_80pct'])
    return results, seg_rows, hier_rows

if __name__ == '__main__':
    main()
