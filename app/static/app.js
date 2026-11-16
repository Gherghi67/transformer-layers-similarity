import {compareMaps, demoMaps} from './math.mjs';

const $ = selector => document.querySelector(selector);
const state = {runs: [], demos: demoMaps(), demo: false, selected: null, tasks: [], view: 'maps', renderVersion: 0};
const metricNames = {cosine: 'Cosine', centered_cosine: 'Mean-centered cosine', linear_cka: 'Linear CKA', js_divergence: 'Jensen–Shannon divergence'};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function message(text, error = false) {
  const box = $('#message');
  box.textContent = text;
  box.classList.toggle('error', error);
  box.hidden = !text;
}

async function api(path, method = 'GET', body) {
  const options = {method};
  if (body !== undefined) {
    options.headers = {'Content-Type': 'application/json'};
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}

function collection() { return state.demo ? state.demos : state.runs; }
async function loadRun(id) {
  if (state.demo) return state.demos.find(run => run.id === id);
  return api(`/api/runs/${id}`);
}

function emptyPanel(title, copy) {
  const panel = element('div', 'panel empty');
  const body = element('div');
  body.append(element('div', 'empty-mark', '▦'), element('h2', '', title), element('p', '', copy));
  panel.append(body);
  return panel;
}

function refreshLibrary() {
  const query = $('#search').value.toLowerCase();
  const kind = $('#kind-filter').value;
  const runs = collection().filter(run => `${run.name} ${run.model} ${run.task} ${run.metric}`.toLowerCase().includes(query) && (!kind || run.kind === kind));
  $('#run-count').textContent = collection().length;
  const list = $('#run-list');
  list.replaceChildren();
  for (const run of runs) {
    const button = element('button', `run-item${state.selected === run.id ? ' selected' : ''}`);
    button.setAttribute('aria-pressed', String(state.selected === run.id));
    button.append(element('strong', '', run.name), element('span', '', `${run.model} · ${run.kind} · ${metricNames[run.metric]}`));
    button.addEventListener('click', () => selectRun(run.id));
    list.append(button);
  }
  if (!runs.length) list.append(element('p', 'subline', collection().length ? 'No matching maps.' : 'Your imported maps will appear here.'));
}

function color(value, min, max, diverging) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  let low, high, blend;
  if (diverging) {
    low = t < 0.5 ? [63, 110, 172] : [248, 246, 237];
    high = t < 0.5 ? [248, 246, 237] : [182, 84, 58];
    blend = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  } else {
    low = t < 0.5 ? [247, 248, 233] : [104, 175, 160];
    high = t < 0.5 ? [104, 175, 160] : [20, 61, 76];
    blend = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  }
  return `rgb(${low.map((v, i) => Math.round(v + (high[i] - v) * blend)).join(',')})`;
}

function heatmap(run, matrix = run.matrix, difference = false) {
  const wrapper = element('div', 'heatmap');
  const canvas = element('canvas');
  canvas.width = 800;
  canvas.height = 800;
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', `${difference ? 'Difference' : metricNames[run.metric]} heatmap, ${run.labels.length} by ${run.labels.length}. Use arrow keys to inspect values.`);
  const readout = element('div', 'cell-value', 'Hover, tap or use arrow keys to inspect a cell');
  readout.setAttribute('aria-live', 'polite');
  const size = matrix.length, margin = 66, plot = 716, cell = plot / size;
  const ctx = canvas.getContext('2d');
  let min = run.metric.includes('cosine') ? -1 : 0, max = 1;
  if (difference) {
    let extent = 0;
    for (const row of matrix) for (const value of row) extent = Math.max(extent, Math.abs(value));
    max = extent || 1;
    min = -max;
  }
  ctx.fillStyle = '#fffefa';
  ctx.fillRect(0, 0, 800, 800);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      ctx.fillStyle = color(matrix[i][j], min, max, difference);
      ctx.fillRect(margin + j * cell, 12 + i * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
  ctx.font = '13px Inter, Arial, sans-serif';
  ctx.fillStyle = '#62717b';
  const step = Math.max(1, Math.ceil(size / 13));
  for (let i = 0; i < size; i += step) {
    ctx.textAlign = 'right';
    ctx.fillText(run.labels[i].slice(0, 9), margin - 8, 12 + (i + 0.5) * cell + 4);
    ctx.save();
    ctx.translate(margin + (i + 0.5) * cell, 741);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(run.labels[i].slice(0, 9), 0, 0);
    ctx.restore();
  }
  let selectedRow = 0, selectedCol = 0;
  function inspect(row, col) {
    selectedRow = row;
    selectedCol = col;
    readout.textContent = `${run.labels[row]} × ${run.labels[col]} = ${matrix[row][col].toFixed(6)}`;
  }
  function pointer(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * 800 / rect.width;
    const y = (event.clientY - rect.top) * 800 / rect.height;
    const row = Math.floor((y - 12) / cell), col = Math.floor((x - margin) / cell);
    if (row >= 0 && row < size && col >= 0 && col < size) inspect(row, col);
  }
  canvas.addEventListener('pointermove', pointer);
  canvas.addEventListener('pointerdown', pointer);
  canvas.addEventListener('keydown', event => {
    const moves = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]};
    if (!moves[event.key]) return;
    event.preventDefault();
    const [dr, dc] = moves[event.key];
    inspect(Math.max(0, Math.min(size - 1, selectedRow + dr)), Math.max(0, Math.min(size - 1, selectedCol + dc)));
  });
  const legend = element('div', 'color-legend');
  legend.append(element('span', '', min.toFixed(difference ? 2 : 0)), element('span', `gradient${difference ? ' diverging' : ''}`), element('span', '', max.toFixed(difference ? 2 : 0)));
  wrapper.append(canvas, readout, legend);
  if (run.metric === 'js_divergence' && !difference) wrapper.append(element('p', 'legend-caption', 'Base-2 Jensen–Shannon · 0 means identical patterns'));
  const download = element('button', 'secondary', 'Download PNG ↓');
  download.addEventListener('click', () => {
    const image = document.createElement('canvas');
    image.width = 900;
    image.height = 1020;
    const context = image.getContext('2d');
    context.fillStyle = '#fffefa';
    context.fillRect(0, 0, image.width, image.height);
    context.fillStyle = '#182b3b';
    context.font = 'bold 22px Arial';
    context.fillText(`${difference ? 'Difference A − B' : run.name}`.slice(0, 65), 50, 42);
    context.font = '14px Arial';
    context.fillText(`${run.demo ? 'SYNTHETIC DEMO · ' : ''}${run.model} · ${run.task}`.slice(0, 100), 50, 69);
    context.fillText(`${difference ? 'A − B' : metricNames[run.metric]} · ${run.pooling} · n=${run.sample_count}`, 50, 91);
    context.drawImage(canvas, 50, 115);
    for (let x = 0; x < 300; x++) {
      context.fillStyle = color(min + (max - min) * x / 299, min, max, difference);
      context.fillRect(300 + x, 938, 1, 14);
    }
    context.fillStyle = '#182b3b';
    context.fillText(min.toFixed(3), 240, 951);
    context.fillText(max.toFixed(3), 615, 951);
    context.fillText('Layer Atlas · rows and columns follow the imported label order', 50, 990);
    image.toBlob(blob => {
      if (!blob) return;
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `${difference ? 'difference-a-minus-b' : 'similarity-map'}${run.demo ? '-synthetic-demo' : ''}.png`;
      link.click();
      track('png.export', {name: run.name, id: run.id, difference, demo: Boolean(run.demo)});
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, 'image/png');
  });
  wrapper.append(download);
  return wrapper;
}

function mapPanel(run, title, matrix, difference = false) {
  const panel = element('div', 'panel map-panel');
  const heading = element('div', 'map-heading');
  const text = element('div');
  text.append(element('h2', '', title || run.name), element('span', 'subline', `${run.task} · ${difference ? 'A − B' : metricNames[run.metric]} · ${run.labels.length} ${run.kind === 'head' ? 'heads' : 'layers'}`));
  heading.append(text, element('span', `badge${run.demo ? ' demo' : ''}`, run.demo ? 'Synthetic demo' : 'Imported result'));
  panel.append(heading, heatmap(run, matrix || run.matrix, difference));
  return panel;
}

function displayValue(value) { return value === null || value === undefined || value === '' ? '—' : String(value); }
function metadata(run) {
  const list = element('dl', 'metadata');
  const entries = [
    ['Model', run.model], ['Task / split', `${run.task} / ${run.split || '—'}`], ['Pooling', run.pooling], ['Examples', run.sample_count],
    ['Max tokens', run.max_tokens], ['Seed', run.seed], ['Accuracy', run.accuracy === null ? null : `${(run.accuracy * 100).toFixed(2)}%`], ['Chance', run.chance_accuracy === null ? null : `${(run.chance_accuracy * 100).toFixed(2)}%`],
  ];
  for (const [label, value] of entries) {
    const item = element('div');
    item.append(element('dt', '', label), element('dd', '', displayValue(value)));
    list.append(item);
  }
  return list;
}

async function selectRun(id) {
  state.selected = id;
  refreshLibrary();
  const version = ++state.renderVersion;
  try {
    const run = await loadRun(id);
    if (version !== state.renderVersion) return;
    const panel = mapPanel(run);
    panel.append(metadata(run));
    for (const warning of run.warnings || []) panel.append(element('p', 'warning', warning));
    if (run.accuracy !== null && run.chance_accuracy !== null && run.accuracy <= run.chance_accuracy) panel.append(element('p', 'warning', 'Measured accuracy is at or below the supplied chance baseline. Consider this when interpreting the map.'));
    if (run.provenance) panel.append(element('p', 'subline', `Provenance · ${run.provenance}`));
    const notes = element('div', 'panel notes-panel');
    const label = element('label', '', 'Run notes');
    label.htmlFor = 'run-notes';
    const textarea = element('textarea');
    textarea.id = 'run-notes';
    textarea.rows = 4;
    textarea.maxLength = 20000;
    textarea.value = run.notes;
    textarea.placeholder = 'Record input formatting, observations or caveats for this run.';
    const actions = element('div', 'notes-actions');
    const downloads = element('div', 'download-actions');
    if (!run.demo) {
      for (const [name, suffix] of [['Download JSON ↓', 'export.json'], ['Matrix CSV ↓', 'matrix.csv']]) {
        const link = element('a', '', name);
        link.href = `/api/runs/${run.id}/${suffix}`;
        downloads.append(link);
      }
      const save = element('button', '', 'Save notes');
      save.addEventListener('click', async () => {
        save.disabled = true;
        try {
          const updated = await api(`/api/runs/${run.id}`, 'PATCH', {notes: textarea.value, revision: run.revision});
          run.revision = updated.revision;
          run.notes = updated.notes;
          message('Notes saved.');
        } catch (error) { message(error.message, true); }
        finally { save.disabled = false; }
      });
      const remove = element('button', 'danger', 'Delete run');
      remove.addEventListener('click', async () => {
        if (!confirm(`Delete “${run.name}”? Download JSON first if you need a copy.`)) return;
        try {
          await api(`/api/runs/${run.id}`, 'DELETE', {});
          state.runs = state.runs.filter(item => item.id !== run.id);
          state.selected = null;
          renderCollection();
          message('Run deleted.');
        } catch (error) { message(error.message, true); }
      });
      actions.append(downloads, remove, save);
    } else {
      textarea.readOnly = true;
      textarea.placeholder = 'Demo maps are temporary. Import your own result to save notes.';
    }
    notes.append(label, textarea, actions);
    $('#map-detail').replaceChildren(panel, notes);
  } catch (error) { message(error.message, true); }
}

function refreshCompareSelectors() {
  for (const selector of ['#compare-a', '#compare-b']) {
    const select = $(selector), previous = select.value;
    select.replaceChildren();
    for (const run of collection()) {
      const option = element('option', '', run.name);
      option.value = run.id;
      select.append(option);
    }
    if (collection().some(run => run.id === previous)) select.value = previous;
  }
  if (!$('#compare-b').value || $('#compare-b').value === $('#compare-a').value) $('#compare-b').selectedIndex = Math.min(1, collection().length - 1);
}

let comparisonVersion = 0;
async function renderComparison() {
  const container = $('#comparison');
  if (collection().length < 2) {
    container.replaceChildren(emptyPanel('Choose two maps', 'Import two results, or explore the synthetic demo maps, to compare representations.'));
    return;
  }
  const version = ++comparisonVersion;
  try {
    const [a, b] = await Promise.all([loadRun($('#compare-a').value), loadRun($('#compare-b').value)]);
    if (version !== comparisonVersion) return;
    track('maps.compare', {a: a.id, b: b.id, demo: state.demo});
    const result = compareMaps(a, b);
    const summary = element('div', 'panel comparison-summary');
    if (!result.compatible) {
      summary.append(element('p', 'summary-copy', result.reason));
    } else {
      summary.append(element('span', 'correlation', result.correlation === null ? 'Undefined' : result.correlation.toFixed(4)));
      const copy = element('div');
      copy.append(element('h3', '', 'Upper-triangle Pearson correlation'), element('p', 'summary-copy', `${result.pairs} off-diagonal pairs. ${result.correlation === null ? 'Requires at least two pairs and nonzero variance in both maps.' : 'The diagonal is excluded.'}`));
      summary.append(copy);
    }
    const maps = element('div', `comparison-maps${result.difference ? '' : ' two'}`);
    maps.append(mapPanel(a, `A · ${a.name}`), mapPanel(b, `B · ${b.name}`));
    if (result.difference) maps.append(mapPanel(a, 'Difference · A − B', result.difference, true));
    container.replaceChildren(summary);
    if (result.compatible && !result.difference) container.append(element('p', 'warning', 'Metrics differ. Correlation compares map structure; a difference map requires matching metrics.'));
    container.append(maps, comparisonContext(a, b));
  } catch (error) { message(error.message, true); }
}

function comparisonContext(a, b) {
  const panel = element('div', 'panel context-table');
  panel.append(element('h2', '', 'Comparison context'));
  const table = element('table'), head = element('thead'), header = element('tr');
  for (const title of ['Setting', 'Map A', 'Map B']) header.append(element('th', '', title));
  head.append(header);
  const body = element('tbody');
  for (const key of ['model', 'task', 'split', 'kind', 'metric', 'pooling', 'sample_count', 'max_tokens', 'seed', 'accuracy', 'chance_accuracy']) {
    const row = element('tr', a[key] !== b[key] ? 'different' : '');
    row.append(element('td', '', key.replaceAll('_', ' ')), element('td', '', displayValue(a[key])), element('td', '', displayValue(b[key])));
    body.append(row);
  }
  table.append(head, body);
  panel.append(table, element('p', 'subline', 'Highlighted settings differ. Pooling, input length, sampling and model changes can affect interpretation.'));
  return panel;
}

function renderCollection() {
  ++state.renderVersion;
  ++comparisonVersion;
  $('#demo-banner').hidden = !state.demo;
  $('#demo-toggle').textContent = state.demo ? 'Back to my results' : 'Explore demo maps';
  refreshLibrary();
  refreshCompareSelectors();
  const runs = collection();
  if (runs.length) selectRun(state.selected && runs.some(run => run.id === state.selected) ? state.selected : runs[0].id);
  else $('#map-detail').replaceChildren(emptyPanel('Your first map starts here', 'Export a computed similarity matrix from Colab, then import its JSON file. Inspect layers and heads, compare tasks and keep notes beside your results.'));
  if (state.view === 'compare') renderComparison();
}

function renderTasks() {
  const complete = state.tasks.filter(task => task.done).length;
  $('#task-progress').textContent = `${complete} / ${state.tasks.length}`;
  $('#progress').value = complete;
  const container = $('#tasks');
  container.replaceChildren();
  for (const task of state.tasks) {
    const row = element('div', `task-row${task.done ? ' done' : ''}`);
    const input = element('input');
    input.type = 'checkbox';
    input.checked = task.done;
    input.id = `task-${task.id}`;
    const label = element('label', '', task.title);
    label.htmlFor = input.id;
    const date = element('time', '', new Date(`${task.due}T12:00:00`).toLocaleDateString('en-GB', {day: '2-digit', month: 'short'}));
    date.dateTime = task.due;
    input.addEventListener('change', async () => {
      input.disabled = true;
      try {
        const updated = await api(`/api/tasks/${task.id}`, 'PATCH', {done: input.checked});
        Object.assign(task, updated);
        renderTasks();
      } catch (error) { input.checked = task.done; message(error.message, true); }
      finally { input.disabled = false; }
    });
    row.append(input, label, date);
    container.append(row);
  }
}

function track(action, details) {
  api('/api/activity', 'POST', {action, details}).catch(error => message(`Activity could not be recorded. ${error.message}`, true));
}

function showIdentity(identity) {
  $('#identity-label').textContent = `${identity.actor} · ${identity.identity_source}`;
  const local = identity.identity_source !== 'tailscale';
  for (const node of $('#identity-form').querySelectorAll('label, input, button')) node.hidden = !local;
  $('#visitor-name').required = local;
  if (identity.identity_source === 'self-reported') $('#visitor-name').value = identity.actor;
}

let oldestActivity = null;
let activityVersion = 0;
async function renderActivity(older = false) {
  const version = ++activityVersion;
  try {
    const events = await api(`/api/activity${older && oldestActivity ? `?before=${oldestActivity}` : ''}`);
    if (version !== activityVersion) return;
    const list = $('#activity-list');
    if (!older) list.replaceChildren();
    if (!events.length && !older) list.append(element('p', '', 'No activity recorded yet.'));
    const labels = {'run.import': 'Imported result', 'run.view': 'Viewed result', 'run.export': 'Downloaded result', 'notes.update': 'Edited notes', 'run.delete': 'Deleted result', 'task.update': 'Changed task', 'request.failed': 'Request failed', 'identity.set': 'Set local name', 'view.open': 'Opened view', 'maps.compare': 'Compared maps', 'png.export': 'Exported PNG', 'demo.toggle': 'Changed demo mode'};
    for (const event of events) {
      const row = element('article', 'activity-row');
      const time = element('time', '', new Date(event.created_at).toLocaleString());
      time.dateTime = event.created_at;
      row.append(time, element('strong', '', `${event.actor} · ${event.identity_source}`), element('span', '', labels[event.action] || event.action));
      const details = {...event.details};
      delete details.browser_reported;
      let description = details.name || details.title || '';
      if (event.action === 'task.update') description += ` · ${details.before ? 'Complete' : 'Pending'} → ${details.done ? 'Complete' : 'Pending'}`;
      if (event.action === 'notes.update') description += ` · Revision ${details.from_revision} → ${details.revision} · ${details.characters} characters`;
      if (event.action === 'run.export') description += ` · ${details.format === 'matrix.csv' ? 'CSV' : 'JSON'}`;
      if (event.action === 'request.failed') description = `${details.method} ${event.target} · HTTP ${details.status}`;
      if (event.action === 'maps.compare') description = `${details.a} ↔ ${details.b}${details.demo ? ' · Demo' : ''}`;
      if (event.action === 'view.open') description = details.view;
      if (event.action === 'demo.toggle') description = details.enabled ? 'Demo enabled' : 'Demo disabled';
      if (event.action === 'png.export' && details.difference) description += ' · Difference A − B';
      const detail = element('span', 'activity-detail', description);
      detail.title = `Reference: ${event.target} · Request: ${event.request_id}`;
      row.append(detail);
      list.append(row);
    }
    if (events.length) oldestActivity = events.at(-1).id;
    $('#older-activity').hidden = events.length < 100;
  } catch (error) { message(`Could not load activity. ${error.message}`, true); }
}

$('#identity-form').addEventListener('submit', async event => {
  event.preventDefault();
  try { showIdentity(await api('/api/me', 'POST', {name: $('#visitor-name').value})); }
  catch (error) { message(error.message, true); }
});
$('#refresh-activity').addEventListener('click', () => renderActivity());
$('#older-activity').addEventListener('click', () => renderActivity(true));

for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => {
    state.view = button.dataset.view;
    for (const tab of document.querySelectorAll('[data-view]')) {
      tab.classList.toggle('active', tab === button);
      tab.setAttribute('aria-pressed', String(tab === button));
    }
    for (const view of ['maps', 'compare', 'tasks', 'activity']) $(`#${view}-view`).hidden = view !== state.view;
    $('#view-title').textContent = {maps: 'Explore representations', compare: 'Compare similarity maps', tasks: 'The road to 03 December', activity: 'Who did what'}[state.view];
    if (state.view === 'compare') renderComparison();
    track('view.open', {view: state.view});
    if (state.view === 'activity') renderActivity();
  });
}
$('#demo-toggle').addEventListener('click', () => { state.demo = !state.demo; state.selected = null; message(''); track('demo.toggle', {enabled: state.demo}); renderCollection(); });
$('#search').addEventListener('input', refreshLibrary);
$('#kind-filter').addEventListener('change', refreshLibrary);
$('#compare-a').addEventListener('change', renderComparison);
$('#compare-b').addEventListener('change', renderComparison);
$('#swap').addEventListener('click', () => {
  const a = $('#compare-a').value;
  $('#compare-a').value = $('#compare-b').value;
  $('#compare-b').value = a;
  renderComparison();
});
$('#import-button').addEventListener('click', () => $('#import-file').click());
$('#import-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  $('#import-button').disabled = true;
  try {
    if (file.size > 25 * 1024 * 1024) throw new Error('Imports may be at most 25 MiB.');
    let payload;
    try { payload = JSON.parse(await file.text()); } catch { throw new Error('Choose a valid JSON result file.'); }
    const run = await api('/api/runs', 'POST', payload);
    state.runs = await api('/api/runs');
    state.demo = false;
    state.selected = run.id;
    renderCollection();
    document.querySelector('[data-view="maps"]').click();
    message(`Imported “${run.name}”.`);
  } catch (error) { message(error.message, true); }
  finally { $('#import-button').disabled = false; event.target.value = ''; }
});

try {
  [state.runs, state.tasks] = await Promise.all([api('/api/runs'), api('/api/tasks')]);
  renderCollection();
  renderTasks();
  showIdentity(await api('/api/me'));
  track('view.open', {view: 'maps'});
} catch (error) {
  message(`Could not load the workbench. ${error.message}`, true);
}
