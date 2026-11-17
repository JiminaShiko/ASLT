// Material records are loaded and changed only through the existing FastAPI application.
let materialPage = 1, materialCategory = '', materialTimer, materialRequest;
const materialPageSize = 9;
let materialState = { items: [], total: 0, categories: [], status: 'loading', error: '', busy: false, editing: null, deleting: null };
function materialImage(value) {
  if (!value) return '';
  try { const url = new URL(value, location.origin); return url.protocol === 'https:' || (url.protocol === 'http:' && url.origin === location.origin) ? url.href : ''; } catch (_) { return ''; }
}
function materialSearch() { materialPage = 1; clearTimeout(materialTimer); materialRequest?.abort(); materialTimer = setTimeout(loadMaterials, 300); }
function changeMaterialCategory(value) { materialCategory = value; materialPage = 1; loadMaterials(); }
function resetMaterialFilters() { state.query = ''; materialCategory = ''; materialPage = 1; document.getElementById('search').value = ''; loadMaterials(); }
function moveMaterialPage(step) { materialPage = Math.max(1, Math.min(Math.max(1, Math.ceil(materialState.total / materialPageSize)), materialPage + step)); loadMaterials(); }
async function loadMaterials() {
  clearTimeout(materialTimer); materialRequest?.abort(); const controller = new AbortController(); materialRequest = controller;
  materialState.status = 'loading'; materialState.items = []; materialState.total = 0; renderContent(); updateConnection();
  try {
    let result = await window.SignrestAPI.materials({ page: materialPage, page_size: materialPageSize, q: state.query, category: materialCategory }, controller.signal);
    if (controller.signal.aborted) return;
    const last = Math.max(1, Math.ceil(result.total / materialPageSize));
    if (materialPage > last) { materialPage = last; return loadMaterials(); }
    Object.assign(materialState, result, { status: 'ready', error: '' });
  } catch (error) { if (controller.signal.aborted) return; materialState.status = 'error'; materialState.error = error.message || 'Unable to load materials.'; }
  renderContent(); updateConnection();
}
function materialsView() {
  const m = materialState, last = Math.max(1, Math.ceil(m.total / materialPageSize)), start = (materialPage - 1) * materialPageSize;
  return '<div class="materials-heading"><h1 class="view-title">Sign Materials</h1><button class="btn" onclick="openMaterialEditor()">' + icon('plus') + 'Add material</button></div>' +
    '<div class="toolbar"><div class="material-filters"><label for="material-category">Category</label><select id="material-category" class="period" onchange="changeMaterialCategory(this.value)"><option value="">All categories</option>' + [...new Set([...m.categories, ...(materialCategory ? [materialCategory] : [])])].map(c => '<option value="' + esc(c) + '" ' + (c === materialCategory ? 'selected' : '') + '>' + esc(c) + '</option>').join('') + '</select><button class="text-button" onclick="resetMaterialFilters()">Reset filters</button></div><button class="text-button" onclick="loadMaterials()">Refresh ' + icon('arrow') + '</button></div>' +
    (m.status !== 'ready' ? statusBlock(m.status, m.error, '', 'loadMaterials') : m.items.length ? '<div class="material-grid">' + m.items.map(item => {
      const image = materialImage(item.image);
      return '<article class="card library-material"><div class="material-media">' + (image ? '<img src="' + esc(image) + '" alt="' + esc(item.label) + ' sign" loading="lazy"><span class="material-image-fallback" hidden>' + icon('hand') + '<span>Image unavailable</span></span>' : icon('hand') + '<span>No image</span>') + '</div><div class="material-body"><span class="badge neutral">' + esc(item.category) + '</span><h2>' + esc(item.label) + '</h2><p>' + esc(item.explanation || 'No explanation added.') + '</p><div class="material-bottom"><button class="text-button" data-id="' + item.id + '" onclick="showMaterialDetails(Number(this.dataset.id))">View details ' + icon('arrow') + '</button><div><button class="icon-button" data-id="' + item.id + '" onclick="openMaterialEditor(Number(this.dataset.id))" aria-label="Edit ' + esc(item.label) + '" title="Edit material">' + icon('edit') + '</button><button class="icon-button" data-id="' + item.id + '" onclick="confirmMaterialDelete(Number(this.dataset.id))" aria-label="Delete ' + esc(item.label) + '" title="Delete material">' + icon('trash') + '</button></div></div></div></article>';
    }).join('') + '</div>' : '<div class="material-empty">' + icon('book') + '<h2>' + (state.query || materialCategory ? 'No matching materials' : 'No materials yet') + '</h2><p>' + (state.query || materialCategory ? 'Try another search or category.' : 'Your library is empty.') + '</p></div>') +
    '<div class="log-pagination"><span>' + (m.items.length ? start + 1 : 0) + '–' + (start + m.items.length) + ' of ' + m.total + ' materials</span><div><button class="icon-button" aria-label="Previous page" onclick="moveMaterialPage(-1)" ' + (materialPage === 1 || m.status !== 'ready' ? 'disabled' : '') + '>' + icon('arrow').replace('<svg ', '<svg class="previous-arrow" ') + '</button><span>Page ' + materialPage + ' of ' + last + '</span><button class="icon-button" aria-label="Next page" onclick="moveMaterialPage(1)" ' + (materialPage >= last || m.status !== 'ready' ? 'disabled' : '') + '>' + icon('arrow') + '</button></div></div>';
}
function materialDialog(title, fields, action) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-fields').innerHTML = fields + '<p id="material-error" class="material-error field full" role="alert" hidden></p>';
  const actions = document.querySelector('#modal-form .form-actions'); actions.hidden = !action;
  document.querySelector('#modal-form button[type="submit"]').textContent = action || 'Save material';
  document.getElementById('modal').showModal();
}
function openMaterialEditor(id) {
  if (materialState.busy) return;
  const item = id == null ? null : materialState.items.find(m => m.id === id); if (id != null && !item) return;
  materialState.editing = item?.id ?? null; materialState.deleting = null;
  materialDialog(item ? 'Edit material' : 'Add material',
    '<label class="field full"><span>Sign label</span><input name="sign_label" required maxlength="100" value="' + esc(item?.label || '') + '"></label>' +
    '<label class="field full"><span>Category</span><input name="category" list="material-categories" required maxlength="30" value="' + esc(item?.category || '') + '"><datalist id="material-categories">' + materialState.categories.map(c => '<option value="' + esc(c) + '">').join('') + '</datalist></label>' +
    '<label class="field full"><span>Image URL (optional)</span><input name="image_path" maxlength="500" value="' + esc(item?.image || '') + '"></label>' +
    '<label class="field full"><span>Explanation</span><textarea name="explanation" maxlength="10000">' + esc(item?.explanation || '') + '</textarea></label>', 'Save material');
}
function showMaterialDetails(id) {
  const item = materialState.items.find(m => m.id === id); if (!item) return;
  const image = materialImage(item.image);
  materialDialog(item.label, (image ? '<div class="material-detail-media field full"><img src="' + esc(image) + '" alt="' + esc(item.label) + ' sign"><span class="material-image-fallback" hidden>Image unavailable</span></div>' : '') + '<dl class="log-details">' + [['Category', item.category], ['Explanation', item.explanation || '—'], ['Created', dateLabel(item.createdAt)]].map(([k,v]) => '<div><dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd></div>').join('') + '</dl>', null);
}
function confirmMaterialDelete(id) {
  if (materialState.busy) return;
  const item = materialState.items.find(m => m.id === id); if (!item) return;
  materialState.deleting = id;
  materialDialog('Delete material?', '<p class="field full">Delete “' + esc(item.label) + '” from the library? This cannot be undone.</p>', 'Delete material');
}
async function saveModal(event) {
  event.preventDefault(); if (state.page !== 'materials' || materialState.busy) return;
  const form = document.getElementById('modal-form'), error = document.getElementById('material-error'); if (!error || !form.reportValidity()) return;
  error.hidden = true;
  let body;
  if (materialState.deleting == null) {
    const data = new FormData(form); body = Object.fromEntries(['sign_label', 'category', 'image_path', 'explanation'].map(k => [k, String(data.get(k) || '').trim()]));
    if (!body.sign_label || !body.category) { error.textContent = 'Enter a sign label and category.'; error.hidden = false; return; }
    if (body.image_path && !materialImage(body.image_path)) { error.textContent = 'Use an HTTPS image URL or a public image path on your application.'; error.hidden = false; return; }
    body.image_path = body.image_path || null;
  }
  materialState.busy = true; form.querySelectorAll('button,input,textarea').forEach(el => el.disabled = true); document.getElementById('modal-close').disabled = true;
  try {
    if (materialState.deleting != null) await window.SignrestAPI.deleteMaterial(materialState.deleting);
    else if (materialState.editing != null) await window.SignrestAPI.updateMaterial(materialState.editing, body);
    else await window.SignrestAPI.createMaterial(body);
    document.getElementById('modal').close(); toast(materialState.deleting != null ? 'Material deleted.' : 'Material saved.'); await loadMaterials();
  } catch (e) { error.textContent = e.message || 'The change could not be saved.'; error.hidden = false; }
  finally { materialState.busy = false; form.querySelectorAll('button,input,textarea').forEach(el => el.disabled = false); document.getElementById('modal-close').disabled = false; }
}
document.addEventListener('error', event => { const img = event.target; if (img instanceof HTMLImageElement && img.closest('.material-media,.material-detail-media')) { img.hidden = true; if (img.nextElementSibling) img.nextElementSibling.hidden = false; } }, true);
document.getElementById('modal').addEventListener('cancel', event => { if (materialState.busy) event.preventDefault(); });