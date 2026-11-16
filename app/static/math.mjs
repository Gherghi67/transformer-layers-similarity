export function compareMaps(a, b) {
  if (a.kind !== b.kind || a.labels.length !== b.labels.length || a.labels.some((label, i) => label !== b.labels[i])) {
    return {compatible: false, reason: 'Maps need the same kind and labels in the same order.'};
  }
  const x = [], y = [];
  for (let i = 0; i < a.matrix.length; i++) {
    for (let j = i + 1; j < a.matrix.length; j++) {
      x.push(a.matrix[i][j]);
      y.push(b.matrix[i][j]);
    }
  }
  const meanX = x.reduce((sum, value) => sum + value, 0) / x.length;
  const meanY = y.reduce((sum, value) => sum + value, 0) / y.length;
  let covariance = 0, varianceX = 0, varianceY = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - meanX, dy = y[i] - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  const constant = x.every(value => value === x[0]) || y.every(value => value === y[0]);
  const correlation = x.length < 2 || constant || varianceX === 0 || varianceY === 0 ? null : Math.max(-1, Math.min(1, covariance / Math.sqrt(varianceX * varianceY)));
  const difference = a.metric === b.metric ? a.matrix.map((row, i) => row.map((value, j) => value - b.matrix[i][j])) : null;
  return {compatible: true, correlation, difference, pairs: x.length};
}

export function demoMaps() {
  const labels = Array.from({length: 24}, (_, i) => `L${i}`);
  function matrix(reasoning, cka) {
    return labels.map((_, i) => labels.map((_, j) => {
      if (i === j) return 1;
      const distance = Math.abs(i - j);
      const middle = i > 1 && i < 22 && j > 1 && j < 22;
      const block = middle ? (reasoning && Math.floor(i / 6) !== Math.floor(j / 6) ? 0.49 : 0.87) : 0.27;
      const value = Math.min(0.98, block + 0.1 * Math.exp(-distance / 3) + 0.012 * Math.cos(i + j));
      return cka ? value ** 2 : value;
    }));
  }
  return [
    {id: 'demo-semantic', name: 'LAMBADA · cosine', task: 'LAMBADA', metric: 'cosine', matrix: matrix(false, false)},
    {id: 'demo-reasoning', name: 'GSM8K · cosine', task: 'GSM8K', metric: 'cosine', matrix: matrix(true, false)},
    {id: 'demo-cka', name: 'LAMBADA · linear CKA', task: 'LAMBADA', metric: 'linear_cka', matrix: matrix(false, true)},
  ].map(run => ({schema_version: 1, model: 'Synthetic 24-layer model', split: 'synthetic', kind: 'layer', pooling: 'masked_mean', sample_count: 400, seed: 42, max_tokens: 256, accuracy: null, chance_accuracy: null, labels, notes: '', provenance: 'Deterministic synthetic data for exploring the interface. These values are invented.', warnings: [], demo: true, ...run}));
}
