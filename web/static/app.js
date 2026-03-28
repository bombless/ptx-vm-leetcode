const problemNumber = document.getElementById("problem-number");
const problemTitle = document.getElementById("problem-title");
const problemMeta = document.getElementById("problem-meta");
const problemStatement = document.getElementById("problem-statement");
const problemSignature = document.getElementById("problem-signature");
const requirementsList = document.getElementById("requirements-list");
const examplesList = document.getElementById("examples-list");
const constraintsList = document.getElementById("constraints-list");
const notesList = document.getElementById("notes-list");
const codeEditor = document.getElementById("code-editor");
const importFile = document.getElementById("import-file");
const resetButton = document.getElementById("reset-button");
const runSamplesButton = document.getElementById("run-samples-button");
const submitButton = document.getElementById("submit-button");
const actionStatus = document.getElementById("action-status");
const autosaveLabel = document.getElementById("autosave-label");
const charCount = document.getElementById("char-count");
const resultModeChip = document.getElementById("result-mode-chip");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultSummary = document.getElementById("result-summary");
const caseResults = document.getElementById("case-results");
const judgeLogs = document.getElementById("judge-logs");

const state = {
  problem: null,
  storageKey: null,
};

function setStatus(message, tone) {
  actionStatus.textContent = message;
  actionStatus.className = `status-banner ${tone}`;
}

function hideStatus() {
  actionStatus.className = "status-banner hidden";
  actionStatus.textContent = "";
}

function renderMeta(problem) {
  problemMeta.innerHTML = "";
  const chips = [
    `#${problem.number}`,
    problem.difficulty,
    problem.category,
  ];

  for (const item of chips) {
    const span = document.createElement("span");
    span.className = "meta-chip";
    span.textContent = item;
    problemMeta.appendChild(span);
  }
}

function renderList(target, items) {
  target.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function renderExamples(examples) {
  examplesList.innerHTML = examples
    .map(
      (example) => `
        <article class="example-card">
          <h4>${example.title}</h4>
          <div class="example-block">
            <span>Input</span>
            <pre>${example.input.join("\n")}</pre>
          </div>
          <div class="example-block">
            <span>Output</span>
            <pre>${example.output}</pre>
          </div>
        </article>
      `
    )
    .join("");
}

function updateCharacterCount() {
  charCount.textContent = `${codeEditor.value.length} chars`;
}

function persistSource() {
  if (!state.storageKey) {
    return;
  }
  localStorage.setItem(state.storageKey, codeEditor.value);
  autosaveLabel.textContent = "Autosaved locally";
  updateCharacterCount();
}

function loadInitialSource(problem) {
  const cached = localStorage.getItem(state.storageKey);
  codeEditor.value = cached || problem.starter_code;
  updateCharacterCount();
}

function renderProblem(problem) {
  problemNumber.textContent = `Problem ${problem.number}`;
  problemTitle.textContent = problem.title;
  renderMeta(problem);
  problemStatement.textContent = problem.statement;
  problemSignature.textContent = problem.signature;
  renderList(requirementsList, problem.implementation_requirements);
  renderExamples(problem.examples);
  renderList(constraintsList, problem.constraints);
  renderList(notesList, problem.notes);
}

function formatArray(values) {
  return `[${values.join(", ")}]`;
}

function renderSummary(payload) {
  const cards = [
    { label: "Status", value: payload.status.replaceAll("_", " ") },
    { label: "Passed", value: `${payload.passed} / ${payload.total}` },
    { label: "Mode", value: payload.mode },
    { label: "Elapsed", value: `${payload.duration_ms} ms` },
  ];

  resultSummary.innerHTML = cards
    .map(
      (card) => `
        <div class="summary-card">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
        </div>
      `
    )
    .join("");
}

function renderCases(cases) {
  caseResults.innerHTML = cases
    .map((item) => {
      const hiddenCopy = item.hidden
        ? `
            <div class="case-brief">
              <span>Length</span>
              <strong>${item.n}</strong>
            </div>
          `
        : `
            <div class="case-grid">
              <div class="case-block">
                <span>Input A</span>
                <pre>${formatArray(item.input.A)}</pre>
              </div>
              <div class="case-block">
                <span>Input B</span>
                <pre>${formatArray(item.input.B)}</pre>
              </div>
              <div class="case-block">
                <span>Expected C</span>
                <pre>${formatArray(item.expected)}</pre>
              </div>
              <div class="case-block">
                <span>Actual C</span>
                <pre>${item.actual ? formatArray(item.actual) : "Unavailable"}</pre>
              </div>
            </div>
          `;

      return `
        <article class="case-card ${item.passed ? "case-pass" : "case-fail"}">
          <div class="case-head">
            <div>
              <h4>${item.name}</h4>
              <p>${item.message}</p>
            </div>
            <span class="case-status">${item.passed ? "Passed" : "Failed"}</span>
          </div>
          ${hiddenCopy}
        </article>
      `;
    })
    .join("");
}

function renderResult(payload) {
  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultModeChip.textContent = payload.mode === "submit" ? "Submit" : "Samples";
  resultModeChip.className = `mode-chip ${payload.ok ? "mode-pass" : "mode-fail"}`;
  renderSummary(payload);
  renderCases(payload.cases || []);
  judgeLogs.textContent = payload.logs || "No runner logs captured.";
}

function setBusy(isBusy, message) {
  runSamplesButton.disabled = isBusy;
  submitButton.disabled = isBusy;
  if (isBusy) {
    setStatus(message, "working");
  }
}

async function loadProblem() {
  setStatus("Loading challenge metadata...", "working");

  try {
    const response = await fetch("/api/problems/gpu-vector-add-f32");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || payload.message || "Failed to load the problem.");
    }

    state.problem = payload.problem;
    state.storageKey = `ptx-arena:${state.problem.id}:source`;
    renderProblem(state.problem);
    loadInitialSource(state.problem);
    setStatus("Challenge loaded. Edit the PTX template and run the samples.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function executeJudge(mode) {
  if (!state.problem) {
    setStatus("The problem is still loading. Please wait a moment.", "error");
    return;
  }

  const source = codeEditor.value.trim();
  if (!source) {
    setStatus("Write or import some PTX code before running the judge.", "error");
    return;
  }

  setBusy(true, mode === "submit" ? "Submitting to the local judge..." : "Running visible sample tests...");
  persistSource();

  try {
    const endpoint =
      mode === "submit"
        ? `/api/problems/${state.problem.id}/submit`
        : `/api/problems/${state.problem.id}/run-samples`;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ source }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || payload.message || "Judge request failed.");
    }

    renderResult(payload);
    setStatus(payload.summary, payload.ok ? "success" : "error");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false, "");
  }
}

codeEditor.addEventListener("input", persistSource);

resetButton.addEventListener("click", () => {
  if (!state.problem) {
    return;
  }
  codeEditor.value = state.problem.starter_code;
  persistSource();
  hideStatus();
});

runSamplesButton.addEventListener("click", () => executeJudge("sample"));
submitButton.addEventListener("click", () => executeJudge("submit"));

importFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const text = await file.text();
  codeEditor.value = text;
  persistSource();
  setStatus(`Loaded ${file.name} into the editor.`, "success");
  importFile.value = "";
});

window.addEventListener("load", loadProblem);
