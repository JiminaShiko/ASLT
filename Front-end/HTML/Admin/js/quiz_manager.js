/* Signrest Admin — Quiz Manager */
(() => {
  const API = "/api/admin";
  const LOW_CONF = 70;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- API layer ---------- */
  async function req(method, path, body) {
    const res = await fetch(API + path, {
      method, credentials: "include",
      headers: { Accept: "application/json", ...(body ? { "Content-Type": "application/json" } : {}), "X-Signrest-Request": "admin" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401) throw new Error("Please sign in again.");
    if (res.status === 403) throw new Error("An administrator account is required.");
    if (!res.ok) {
      if ((res.headers.get("content-type") || "").includes("application/json")) {
        const error = await res.json();
        if (error && typeof error.detail === "string") throw new Error(error.detail);
        if (error && typeof error.message === "string") throw new Error(error.message);
      }
      throw new Error(`The server could not complete this request (HTTP ${res.status}).`);
    }
    return res.status === 204 ? null : res.json();
  }
  const api = {
    async signs() {
      const d = await req("GET", "/quiz/signs");
      return d.items.map((m) => ({ id: m.material_id, label: m.sign_label, category: m.category, media: m.media_url || "", conf: m.model_confidence ?? null }));
    },
    async list() {
      return (await req("GET", "/quiz/questions")).items;
    },
    async save(q) {
      return q.id ? req("PATCH", "/quiz/questions/" + q.id, q) : req("POST", "/quiz/questions", q);
    },
    async remove(id) {
      return req("DELETE", "/quiz/questions/" + id);
    },
    async distractors(materialId) {
      return (await req("GET", "/quiz/distractors?material_id=" + materialId)).items;
    },
  };

  /* ---------- State ---------- */
  let signs = [], questions = [], editing = null, deleting = null;
  let form = { type: "sign_to_text", options: [null, null, null, null], correct: 0 };
  const signById = (id) => signs.find((s) => s.id === Number(id));

  function media(sign, small) {
    if (!sign) return "?";
    if (sign.media && /\.(mp4|webm)$/i.test(sign.media)) return `<video src="${esc(sign.media)}" muted ${small ? "" : "controls"} playsinline></video>`;
    if (sign.media) return `<img src="${esc(sign.media)}" alt="${esc(sign.label)} sign">`;
    return `🤟<br>${esc(sign.label)}`;
  }
  function confBadge(sign) {
    if (!sign || sign.conf == null) return "";
    return `<span class="q-badge ${sign.conf < LOW_CONF ? "low" : "high"}">Model ${sign.conf.toFixed(1)}%</span>`;
  }

  /* ---------- Render list ---------- */
  function render() {
    const q = $("search").value.trim().toLowerCase(), t = $("fType").value, c = $("fCat").value, st = $("fStatus").value;
    const rows = questions.filter((x) => {
      const ans = signById(x.correct)?.label || "";
      const prompt = x.type === "text_to_sign" ? x.prompt_text : signById(x.prompt_material_id)?.label || "";
      return (!t || x.type === t) && (!c || x.category === c) && (!st || x.status === st) && (!q || (prompt + " " + ans).toLowerCase().includes(q));
    });
    $("stTotal").textContent = questions.length;
    $("stS2T").textContent = questions.filter((x) => x.type === "sign_to_text").length;
    $("stT2S").textContent = questions.filter((x) => x.type === "text_to_sign").length;
    $("stLow").textContent = questions.filter((x) => (signById(x.correct)?.conf ?? 100) < LOW_CONF).length;

    if (!rows.length) { $("list").innerHTML = `<div class="q-empty">${questions.length ? "No matching questions." : "No questions yet. Click “Add question”."}</div>`; return; }
    $("list").innerHTML = rows.map((x) => {
      const correct = signById(x.correct), prompt = signById(x.prompt_material_id);
      const isS2T = x.type === "sign_to_text";
      const title = isS2T ? "What does this sign mean?" : `Which sign means “${esc(x.prompt_text)}”?`;
      const optLabels = x.options.map((id) => esc(signById(id)?.label || "—")).join(" · ");
      return `<article class="q-card">
        <div class="q-thumb">${isS2T ? media(prompt, true) : esc(x.prompt_text)}</div>
        <div>
          <div class="q-badges"><span class="q-badge type">${isS2T ? "Sign → Text" : "Text → Sign"}</span><span class="q-badge">${esc(x.category)}</span>${x.status === "draft" ? '<span class="q-badge draft">Draft</span>' : ""}${confBadge(correct)}</div>
          <h3>${title}</h3>
          <p>Answer: <b>${esc(correct?.label || "—")}</b> &nbsp;|&nbsp; Options: ${optLabels}</p>
        </div>
        <div class="q-row-actions">
          <button class="q-icon" data-edit="${x.id}" aria-label="Edit">✎</button>
          <button class="q-icon" data-del="${x.id}" aria-label="Delete">🗑</button>
        </div></article>`;
    }).join("");
  }

  /* ---------- Dialog ---------- */
  const signOptions = (sel) => `<option value="">Choose a sign…</option>` + signs.map((s) => `<option value="${s.id}" ${s.id === Number(sel) ? "selected" : ""}>${esc(s.label)} (${esc(s.category)})</option>`).join("");

  function renderForm() {
    document.querySelectorAll("#typeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.type === form.type));
    $("promptTextWrap").hidden = form.type !== "text_to_sign";
    $("opts").innerHTML = form.options.map((id, i) => `
      <div class="q-opt ${i === form.correct ? "correct" : ""}">
        <input type="radio" name="correct" value="${i}" ${i === form.correct ? "checked" : ""} aria-label="Mark option ${i + 1} correct">
        <select data-opt="${i}" aria-label="Option ${i + 1}">${signOptions(id)}</select>
      </div>`).join("");
    $("promptSignWrap").firstChild.textContent = form.type === "sign_to_text" ? "Sign shown to learner (from dataset)" : "Correct sign for this text (from dataset)";
    updateModel();
  }
  function updateModel() {
    const s = signById($("promptSign").value);
    $("modelBox").innerHTML = s
      ? `Model check for <b>${esc(s.label)}</b>: ${s.conf == null ? "not evaluated yet" : `<b>${s.conf.toFixed(1)}%</b> confidence`}${s.conf != null && s.conf < LOW_CONF ? " — ⚠ the model struggles with this sign; consider another clip." : ""}`
      : "Model check: choose a sign to see the model confidence.";
  }
  function syncPromptToCorrect() {
    // The prompt sign is always the correct answer.
    const id = Number($("promptSign").value) || null;
    form.options[form.correct] = id;
    if (form.type === "text_to_sign" && !$("promptText").value && id) $("promptText").value = signById(id).label;
  }

  function openEditor(q) {
    editing = q?.id ?? null;
    $("dlgTitle").textContent = q ? "Edit question" : "Add question";
    const cats = [...new Set(signs.map((s) => s.category))];
    $("qCat").innerHTML = cats.map((c) => `<option ${c === q?.category ? "selected" : ""}>${esc(c)}</option>`).join("");
    $("qStatus").value = q?.status || "active";
    $("promptSign").innerHTML = signOptions(q?.prompt_material_id);
    $("promptText").value = q?.prompt_text || "";
    form = q ? { type: q.type, options: [...q.options], correct: Math.max(0, q.options.indexOf(q.correct)) } : { type: "sign_to_text", options: [null, null, null, null], correct: 0 };
    $("formError").hidden = true;
    renderForm();
    $("qDialog").showModal();
  }

  async function submit(e) {
    e.preventDefault();
    syncPromptToCorrect();
    const err = (m) => { $("formError").textContent = m; $("formError").hidden = false; };
    const promptId = Number($("promptSign").value);
    if (!promptId) return err("Choose a sign from the dataset.");
    if (form.type === "text_to_sign" && !$("promptText").value.trim()) return err("Enter the text shown to the learner.");
    if (form.options.some((o) => !o)) return err("Fill all four answer options.");
    if (new Set(form.options).size !== 4) return err("Each option must be a different sign.");
    const q = {
      ...(editing ? { id: editing } : {}),
      type: form.type, prompt_material_id: promptId,
      prompt_text: form.type === "text_to_sign" ? $("promptText").value.trim() : "",
      category: $("qCat").value, status: $("qStatus").value,
      options: form.options.map(Number), correct: form.options[form.correct],
    };
    $("saveBtn").disabled = true;
    try { await api.save(q); $("qDialog").close(); toast(editing ? "Question updated." : "Question added."); await load(); }
    catch (x) { err(x.message); }
    finally { $("saveBtn").disabled = false; }
  }

  function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); }

  async function load() {
    try { questions = await api.list(); render(); }
    catch (e) { $("list").innerHTML = `<div class="q-empty">${esc(e.message)}</div>`; }
  }

  /* ---------- Events ---------- */
  document.addEventListener("DOMContentLoaded", async () => {
    const saved = localStorage.getItem("signrest-admin-theme");
    document.documentElement.dataset.theme = saved === "dark" ? "dark" : "light";
    ["search", "fType", "fCat", "fStatus"].forEach((id) => $(id).addEventListener("input", render));
    $("addBtn").onclick = () => openEditor(null);
    $("dlgClose").onclick = $("cancelBtn").onclick = () => $("qDialog").close();
    $("qForm").addEventListener("submit", submit);
    $("typeSeg").onclick = (e) => { const b = e.target.closest("button"); if (b) { form.type = b.dataset.type; renderForm(); } };
    $("promptSign").onchange = () => { const s = signById($("promptSign").value); if (s) $("qCat").value = s.category; syncPromptToCorrect(); renderForm(); };
    $("opts").addEventListener("change", (e) => {
      if (e.target.name === "correct") { form.correct = Number(e.target.value); form.options[form.correct] = Number($("promptSign").value) || form.options[form.correct]; }
      if (e.target.dataset.opt != null) form.options[Number(e.target.dataset.opt)] = Number(e.target.value) || null;
      renderForm();
    });
    $("suggestBtn").onclick = async () => {
      const id = Number($("promptSign").value);
      if (!id) return toast("Choose the sign first.");
      try {
        const ds = (await api.distractors(id)).filter((d) => d !== id);
        let k = 0;
        form.options = form.options.map((o, i) => (i === form.correct ? id : ds[k++] ?? o));
        renderForm(); toast("Distractors suggested by the model.");
      } catch (error) {
        toast(error.message);
      }
    };
    $("list").onclick = (e) => {
      const ed = e.target.closest("[data-edit]"), dl = e.target.closest("[data-del]");
      if (ed) openEditor(questions.find((q) => q.id === Number(ed.dataset.edit)));
      if (dl) {
        deleting = Number(dl.dataset.del);
        $("delText").textContent = "This admin question will be permanently deleted.";
        $("delDialog").showModal();
      }
    };
    $("delCancel").onclick = () => $("delDialog").close();
    $("delOk").onclick = async () => { try { await api.remove(deleting); $("delDialog").close(); toast("Question deleted."); load(); } catch (x) { toast(x.message); } };

    try {
      signs = await api.signs();
      $("fCat").innerHTML += [...new Set(signs.map((s) => s.category))].map((c) => `<option>${esc(c)}</option>`).join("");
    } catch (e) { toast(e.message); }
    load();
  });
})();