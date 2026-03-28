const state = {
  programId: null,
  program: null,
  kernels: [],
};

const inspectForm = document.getElementById("inspect-form");
const inspectButton = document.getElementById("inspect-button");
const inspectStatus = document.getElementById("inspect-status");
const programSummary = document.getElementById("program-summary");
const runCard = document.getElementById("run-card");
const kernelSelect = document.getElementById("kernel-select");
const parameterPanel = document.getElementById("parameter-panel");
const runButton = document.getElementById("run-button");
const runStatus = document.getElementById("run-status");
const resultsCard = document.getElementById("results-card");
const resultSummary = document.getElementById("result-summary");
const scalarResults = document.getElementById("scalar-results");
const pointerResults = document.getElementById("pointer-results");
const watchResults = document.getElementById("watch-results");
const runnerLogs = document.getElementById("runner-logs");

function showStatus(target, tone, message) {
  target.textContent = message;
  target.className = `status-panel ${tone}`;
}

function hideStatus(target) {
  target.className = "status-panel hidden";
  target.textContent = "";
}

function currentKernel() {
  return state.kernels.find((kernel) => kernel.name === kernelSelect.value) || null;
}

function renderProgramSummary(program) {
  programSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-item">
        <span>文件</span>
        <strong>${program.filename}</strong>
      </div>
      <div class="summary-item">
        <span>PTX 版本</span>
        <strong>${program.version || "未声明"}</strong>
      </div>
      <div class="summary-item">
        <span>目标架构</span>
        <strong>${program.target || "未声明"}</strong>
      </div>
      <div class="summary-item">
        <span>地址位数</span>
        <strong>${program.address_size}</strong>
      </div>
    </div>
  `;
  programSummary.classList.remove("hidden");
}

function renderParameters() {
  const kernel = currentKernel();
  if (!kernel) {
    parameterPanel.innerHTML = "";
    return;
  }

  if (!kernel.parameters.length) {
    parameterPanel.innerHTML = `
      <div class="empty-state">
        <strong>${kernel.name}</strong>
        <p>这个 kernel 没有参数，直接点击“运行 PTX”即可。</p>
      </div>
    `;
    return;
  }

  parameterPanel.innerHTML = kernel.parameters
    .map((parameter, index) => {
      if (parameter.is_pointer) {
        return `
          <div class="parameter-card pointer-card" data-kind="pointer" data-name="${parameter.name}">
            <div class="parameter-head">
              <h3>${index + 1}. ${parameter.name}</h3>
              <span>${parameter.type} / pointer</span>
            </div>
            <div class="grid-two">
              <label class="field">
                <span>缓冲区类型</span>
                <select class="pointer-buffer-type">
                  <option value="int32">int32</option>
                  <option value="uint32">uint32</option>
                  <option value="float32">float32</option>
                  <option value="int64">int64</option>
                  <option value="uint64">uint64</option>
                  <option value="float64">float64</option>
                  <option value="bytes">bytes</option>
                </select>
              </label>
              <label class="field">
                <span>元素个数</span>
                <input class="pointer-count" type="number" min="1" value="8" />
              </label>
            </div>
            <label class="field">
              <span>初始值（逗号分隔，可留空）</span>
              <textarea class="pointer-values" rows="3" placeholder="例如：1,2,3,4"></textarea>
            </label>
          </div>
        `;
      }

      return `
        <div class="parameter-card" data-kind="scalar" data-name="${parameter.name}">
          <div class="parameter-head">
            <h3>${index + 1}. ${parameter.name}</h3>
            <span>${parameter.type}</span>
          </div>
          <label class="field">
            <span>标量值</span>
            <input class="scalar-value" type="text" placeholder="输入 ${parameter.type} 对应的值" />
          </label>
        </div>
      `;
    })
    .join("");
}

function buildRunRequest() {
  const kernel = currentKernel();
  if (!kernel) {
    throw new Error("请先选择一个 kernel");
  }

  const scalars = {};
  const pointers = {};

  for (const node of parameterPanel.querySelectorAll(".parameter-card")) {
    const name = node.dataset.name;
    const kind = node.dataset.kind;
    if (kind === "scalar") {
      const input = node.querySelector(".scalar-value");
      const value = input.value.trim();
      if (!value) {
        throw new Error(`请填写标量参数 ${name}`);
      }
      scalars[name] = value;
      continue;
    }

    const bufferType = node.querySelector(".pointer-buffer-type").value;
    const elementCount = Number(node.querySelector(".pointer-count").value);
    if (!Number.isInteger(elementCount) || elementCount <= 0) {
      throw new Error(`指针参数 ${name} 的元素个数必须大于 0`);
    }
    const values = node.querySelector(".pointer-values").value.trim();
    pointers[name] = {
      buffer_type: bufferType,
      element_count: elementCount,
      values,
    };
  }

  const readDimension = (id) => {
    const value = Number(document.getElementById(id).value);
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${id} 必须是大于 0 的整数`);
    }
    return value;
  };

  return {
    kernel: kernel.name,
    grid: [readDimension("grid-x"), readDimension("grid-y"), readDimension("grid-z")],
    block: [readDimension("block-x"), readDimension("block-y"), readDimension("block-z")],
    scalars,
    pointers,
  };
}

function renderScalarResults(scalars) {
  if (!scalars.length) {
    scalarResults.innerHTML = `
      <h3>标量参数</h3>
      <div class="empty-state"><p>这个 kernel 没有标量参数。</p></div>
    `;
    return;
  }

  scalarResults.innerHTML = `
    <h3>标量参数</h3>
    <div class="chip-row">
      ${scalars
        .map(
          (item) => `
            <div class="result-chip">
              <span>${item.name}</span>
              <strong>${item.value}</strong>
              <small>${item.type}</small>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderPointerResults(buffers) {
  if (!buffers.length) {
    pointerResults.innerHTML = `
      <h3>指针缓冲区</h3>
      <div class="empty-state"><p>这个 kernel 没有指针参数。</p></div>
    `;
    return;
  }

  pointerResults.innerHTML = `
    <h3>指针缓冲区</h3>
    <div class="buffer-grid">
      ${buffers
        .map(
          (buffer) => `
            <article class="buffer-card">
              <div class="buffer-head">
                <strong>${buffer.name}</strong>
                <span>${buffer.buffer_type} · ${buffer.device_address}</span>
              </div>
              <p class="mono subtle">
                ${buffer.element_count} elements / ${buffer.byte_size} bytes${
                  buffer.truncated ? ` · 仅展示前 ${buffer.preview_count} 项` : ""
                }
              </p>
              <div class="buffer-section">
                <span>执行前</span>
                <pre>${JSON.stringify(buffer.before, null, 2)}</pre>
                <code>${buffer.hex_before}</code>
              </div>
              <div class="buffer-section">
                <span>执行后</span>
                <pre>${JSON.stringify(buffer.after, null, 2)}</pre>
                <code>${buffer.hex_after}</code>
              </div>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function renderWatchResults(watches) {
  if (!watches.length) {
    watchResults.innerHTML = `
      <h3>固定地址观察</h3>
      <div class="empty-state"><p>没有可展示的固定地址内存预览。</p></div>
    `;
    return;
  }

  watchResults.innerHTML = `
    <h3>固定地址观察</h3>
    <div class="buffer-grid">
      ${watches
        .map(
          (watch) => `
            <article class="buffer-card compact-card">
              <div class="buffer-head">
                <strong>${watch.address}</strong>
                <span>${watch.byte_size} bytes</span>
              </div>
              <div class="buffer-section">
                <span>Hex</span>
                <code>${watch.hex}</code>
              </div>
              <div class="buffer-section">
                <span>int32 预览</span>
                <pre>${JSON.stringify(watch.int32_preview, null, 2)}</pre>
              </div>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function renderRunResults(payload) {
  resultsCard.classList.remove("hidden");
  resultSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-item">
        <span>Kernel</span>
        <strong>${payload.kernel}</strong>
      </div>
      <div class="summary-item">
        <span>Grid</span>
        <strong>${payload.grid.join(" x ")}</strong>
      </div>
      <div class="summary-item">
        <span>Block</span>
        <strong>${payload.block.join(" x ")}</strong>
      </div>
      <div class="summary-item">
        <span>指针缓冲区</span>
        <strong>${payload.pointer_buffers.length}</strong>
      </div>
    </div>
  `;
  renderScalarResults(payload.scalars || []);
  renderPointerResults(payload.pointer_buffers || []);
  renderWatchResults(payload.memory_watch || []);
  runnerLogs.textContent = payload.logs || "没有额外日志输出。";
}

inspectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(inspectForm);
  hideStatus(runStatus);
  resultsCard.classList.add("hidden");
  inspectButton.disabled = true;
  showStatus(inspectStatus, "working", "正在上传并分析 PTX，请稍候...");

  try {
    const response = await fetch("/api/programs/inspect", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail?.error || payload.detail || "PTX 分析失败");
    }

    state.programId = payload.program_id;
    state.program = payload.program;
    state.kernels = payload.kernels || [];

    if (!state.kernels.length) {
      throw new Error("没有在这个 PTX 文件中找到可运行的 entry kernel");
    }

    renderProgramSummary(payload.program);
    kernelSelect.innerHTML = state.kernels
      .map((kernel) => `<option value="${kernel.name}">${kernel.name}</option>`)
      .join("");
    renderParameters();
    runCard.classList.remove("hidden");
    showStatus(
      inspectStatus,
      "success",
      `已分析 ${payload.program.filename}，共找到 ${state.kernels.length} 个 entry kernel。`
    );
  } catch (error) {
    showStatus(inspectStatus, "error", error.message);
  } finally {
    inspectButton.disabled = false;
  }
});

kernelSelect.addEventListener("change", () => {
  renderParameters();
  resultsCard.classList.add("hidden");
  hideStatus(runStatus);
});

runButton.addEventListener("click", async () => {
  if (!state.programId) {
    showStatus(runStatus, "error", "请先上传并分析一个 PTX 文件。");
    return;
  }

  let requestBody;
  try {
    requestBody = buildRunRequest();
  } catch (error) {
    showStatus(runStatus, "error", error.message);
    return;
  }

  runButton.disabled = true;
  resultsCard.classList.add("hidden");
  showStatus(runStatus, "working", "正在调用 PTX VM 执行 kernel...");

  try {
    const response = await fetch(`/api/programs/${state.programId}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });
    const payload = await response.json();

    if (!response.ok) {
      const detail = payload.detail || {};
      throw new Error(detail.error || detail.message || "PTX 执行失败");
    }

    renderRunResults(payload);
    showStatus(runStatus, "success", "执行完成，结果已经展示在下方。");
  } catch (error) {
    showStatus(runStatus, "error", error.message);
  } finally {
    runButton.disabled = false;
  }
});
