/* Draft-only builder for the shared quiz tables. */
(() => {
  const byId = (id) => document.getElementById(id);
  const dialog = byId("managedQuizDialog");
  if (!dialog) return;
  const list = byId("managedQuizList");
  const editor = byId("managedQuizEditor");
  const questionsEl = byId("managedQuizQuestions");
  const save = byId("managedQuizSave");
  const publish = byId("managedQuizPublish");
  const message = byId("managedQuizMessage");
  let quiz = null;

  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const choice = () => ({kind: "multiple_choice", text: "", media_type: "text", media_url: "", options: Array.from({length: 4}, () => ({type: "text", value: ""})), correct: 0});
  const camera = () => ({kind: "camera", text: "", media_type: "text", media_url: "", target_category: "asl", target_item_key: ""});
  const types = (selected) => ["text", "image", "video"].map((type) => `<option value="${type}" ${type === selected ? "selected" : ""}>${type}</option>`).join("");
  const uploadControl = (target, type) => `<span><button type="button" class="q-btn ghost" data-upload="${target}">Upload ${type}</button><input type="file" hidden data-upload-file="${target}" accept="${type === "image" ? "image/png,image/jpeg,image/webp,image/gif" : "video/mp4,video/webm"}"></span>`;

  async function request(path, method = "GET", body) {
    const response = await fetch(`/api/admin/quiz/managed-quizzes${path}`, {
      method, credentials: "same-origin", headers: {"Accept":"application/json", "X-Signrest-Request":"admin", ...(body ? {"Content-Type":"application/json"} : {})},
      body: body ? JSON.stringify(body) : undefined
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status}).`);
    return data;
  }

  function renderQuestions() {
    questionsEl.innerHTML = quiz.questions.map((q, index) => `
      <article class="builder-question" data-question="${index}">
        <div class="q-actions"><h3>${index + 1}. ${q.kind === "camera" ? "Live camera" : "Multiple choice"}</h3><button type="button" class="q-btn ghost" data-remove="${index}">Remove</button></div>
        <label>Question text<input data-field="text" value="${esc(q.text)}" maxlength="4000"></label>
        <label>Question media type<select data-field="media_type">${types(q.media_type)}</select></label>
        ${q.media_type !== "text" ? `<label>Question media URL<input data-field="media_url" value="${esc(q.media_url)}" maxlength="500" placeholder="https://…"></label>${uploadControl("prompt", q.media_type)}` : ""}
        ${q.kind === "camera" ? `
          <label>Model category<select data-field="target_category"><option value="asl" ${q.target_category === "asl" ? "selected" : ""}>Alphabet</option><option value="wlasl" ${q.target_category === "wlasl" ? "selected" : ""}>Word</option></select></label>
          <label>Target letter or supported word<input data-field="target_item_key" value="${esc(q.target_item_key)}" maxlength="100"></label>` : `
          <div>Four answer options — select the correct one</div>
          ${q.options.map((option, n) => `<div class="builder-option"><select data-option-type="${n}" aria-label="Option ${n + 1} type">${types(option.type)}</select><div><input data-option-value="${n}" value="${esc(option.value)}" maxlength="500" placeholder="Text or media URL" aria-label="Option ${n + 1}">${option.type !== "text" ? uploadControl(String(n), option.type) : ""}</div><label><input type="radio" name="correct-${index}" data-correct="${n}" ${q.correct === n ? "checked" : ""}> Correct</label></div>`).join("")}`}
      </article>`).join("");
  }

  async function loadList() {
    list.textContent = "Loading quizzes…";
    try {
      const data = await request("");
      list.replaceChildren();
      if (!data.items.length) { list.textContent = "No quizzes yet."; return; }
      for (const item of data.items) {
        const row = document.createElement("div");
        row.className = "builder-row";
        row.dataset.title = item.title.toLowerCase();
        const text = document.createElement("span");
        text.textContent = `${item.title} · ${item.question_count} questions · ${item.status}`;
        const button = document.createElement("button");
        button.type = "button"; button.className = "q-btn ghost"; button.textContent = "Edit quiz";
        button.addEventListener("click", async () => {
          button.disabled = true; button.setAttribute("aria-busy", "true"); button.innerHTML = '<span class="builder-spinner" aria-hidden="true"></span>Loading';
          try { const data = await request(`/${item.quiz_id}`); openEditor(data.item); dialog.showModal(); }
          catch (error) { list.textContent = error.message; }
          finally { button.disabled = false; button.removeAttribute("aria-busy"); button.textContent = "Edit quiz"; }
        });
        row.append(text, button);
        row.hidden = !row.dataset.title.includes(byId("search").value.trim().toLowerCase());
        list.appendChild(row);
      }
    } catch (error) { list.textContent = error.message; }
  }

  function openEditor(item) {
    quiz = item;
    byId("managedQuizName").value = item.title || "";
    byId("managedQuizDescription").value = item.description || "";
    message.textContent = "";
    editor.hidden = false; save.hidden = false; publish.hidden = false;
    renderQuestions();
  }

  byId("managedBuilderOpen").addEventListener("click", () => { openEditor({title:"", description:"", status:"draft", questions:[]}); dialog.showModal(); });
  byId("search").addEventListener("input", () => {
    const query = byId("search").value.trim().toLowerCase();
    list.querySelectorAll(".builder-row").forEach((row) => { row.hidden = !row.dataset.title.includes(query); });
  });
  byId("managedQuizClose").addEventListener("click", () => dialog.close());
  byId("managedQuizCancel").addEventListener("click", () => dialog.close());
  byId("managedQuizNew").addEventListener("click", () => openEditor({title:"", description:"", status:"draft", questions:[]}));
  byId("managedAddChoice").addEventListener("click", () => { quiz.questions.push(choice()); renderQuestions(); });
  byId("managedAddCamera").addEventListener("click", () => { quiz.questions.push(camera()); renderQuestions(); });
  questionsEl.addEventListener("input", (event) => {
    const card = event.target.closest("[data-question]");
    if (!card || !quiz) return;
    const q = quiz.questions[Number(card.dataset.question)];
    const field = event.target.dataset.field;
    if (field) q[field] = event.target.value;
    if (event.target.dataset.optionValue !== undefined) q.options[Number(event.target.dataset.optionValue)].value = event.target.value;
    if (event.target.dataset.optionType !== undefined) q.options[Number(event.target.dataset.optionType)].type = event.target.value;
    if (event.target.dataset.correct !== undefined) q.correct = Number(event.target.dataset.correct);
    if (field === "media_type" || event.target.dataset.optionType !== undefined) renderQuestions();
  });
  questionsEl.addEventListener("click", (event) => {
    const upload = event.target.closest("[data-upload]");
    if (upload) { upload.parentElement.querySelector("[data-upload-file]").click(); return; }
    const button = event.target.closest("[data-remove]");
    if (button && quiz) { quiz.questions.splice(Number(button.dataset.remove), 1); renderQuestions(); }
  });
  questionsEl.addEventListener("change", async (event) => {
    const input = event.target.closest("[data-upload-file]");
    if (!input || !input.files.length || !quiz) return;
    const card = input.closest("[data-question]");
    const question = quiz.questions[Number(card.dataset.question)];
    const target = input.dataset.uploadFile;
    const button = input.parentElement.querySelector("[data-upload]");
    const previous = button.textContent;
    const form = new FormData();
    form.append("file", input.files[0]);
    save.disabled = true; publish.disabled = true; button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.innerHTML = '<span class="builder-spinner" aria-hidden="true"></span>Uploading';
    message.textContent = "";
    try {
      const response = await fetch("/api/admin/quiz/media", { method:"POST", credentials:"same-origin", headers:{"X-Signrest-Request":"admin"}, body:form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Upload failed.");
      if (!quiz.questions.includes(question)) return;
      if (target === "prompt") {
        if (question.media_type !== data.type) throw new Error("The uploaded file does not match the selected media type.");
        question.media_url = data.url;
      } else {
        const option = question.options[Number(target)];
        if (!option || option.type !== data.type) throw new Error("The uploaded file does not match the selected media type.");
        option.value = data.url;
      }
      renderQuestions();
    } catch (error) { message.textContent = error.message; }
    finally { save.disabled = false; publish.disabled = false; button.disabled = false; button.removeAttribute("aria-busy"); button.textContent = previous; }
  });
  async function saveQuiz(status, button) {
    if (!quiz) return;
    quiz.title = byId("managedQuizName").value;
    quiz.description = byId("managedQuizDescription").value;
    quiz.status = status;
    save.disabled = true; publish.disabled = true; button.setAttribute("aria-busy", "true");
    button.innerHTML = '<span class="builder-spinner" aria-hidden="true"></span>Saving';
    message.textContent = "";
    try {
      const result = await request(quiz.quiz_id ? `/${quiz.quiz_id}` : "", quiz.quiz_id ? "PUT" : "POST", quiz);
      quiz.quiz_id = result.quiz_id;
      message.textContent = status === "active" ? "Quiz published." : "Draft saved.";
      await loadList();
      if (status === "active") dialog.close();
    } catch (error) { message.textContent = error.message; }
    finally { save.disabled = false; publish.disabled = false; button.removeAttribute("aria-busy"); save.textContent = "Save draft"; publish.textContent = "Publish quiz"; }
  }
  save.addEventListener("click", () => saveQuiz("draft", save));
  publish.addEventListener("click", () => saveQuiz("active", publish));
  loadList();
})();
