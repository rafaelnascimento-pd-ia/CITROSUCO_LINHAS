const state = {
    mode: "train",
    currentStage: "step-mode",
    task: null,
    statusTimer: null
};

const stepButtons = {};
const panels = {};

function $(selector) {
    return document.querySelector(selector);
}

function showStep(stepId) {
    Object.values(panels).forEach(panel => panel.classList.remove("active"));
    const panel = panels[stepId];
    if (panel) {
        panel.classList.add("active");
    }
    Object.values(stepButtons).forEach(btn => btn.classList.remove("active"));
    const menuButton = stepButtons[stepId];
    if (menuButton) {
        menuButton.classList.add("active");
    }
    state.currentStage = stepId;
    updateStatusSummary(`Etapa: ${stepId.replace("step-", "")}`);
}

function updateStatusSummary(text) {
    const el = $("#status-summary");
    if (el) {
        el.textContent = text;
    }
}

function setIndicator(active) {
    const dot = $("#status-indicator");
    if (dot) {
        dot.classList.toggle("active", active);
    }
}

function initUI() {
    ["step-mode","step-data","step-params","step-execute","step-result"].forEach(id => {
        stepButtons[id] = document.querySelector(`[data-step="${id}"]`);
        panels[id] = document.getElementById(id);
    });

    $("#btn-next-mode").addEventListener("click", () => {
        state.mode = document.querySelector('input[name="mode"]:checked').value;
        showStep("step-data");
    });
    $("#btn-back-data").addEventListener("click", () => showStep("step-mode"));
    $("#btn-next-data").addEventListener("click", () => showStep("step-params"));
    $("#btn-back-params").addEventListener("click", () => showStep("step-data"));
    $("#btn-next-params").addEventListener("click", () => showStep("step-execute"));
    $("#btn-back-execute").addEventListener("click", () => showStep("step-params"));
    $("#btn-back-result").addEventListener("click", () => showStep("step-execute"));
    $("#btn-start-action").addEventListener("click", () => startAction());

    setupUpload("#upload-talhao", "#input-file-talhao", "talhao", "#filename-talhao");
    setupUpload("#upload-linhas", "#input-file-linhas", "linhas", "#filename-linhas");
    setupUpload("#upload-path-linhas", "#input-file-path-linhas", "path_linhas", "#filename-path-linhas");
    setupUpload("#upload-checkpoint", "#input-file-checkpoint", "checkpoint", "#filename-checkpoint");
    setupUpload("#upload-mask", "#input-file-mask", "mask", "#filename-mask");

    // Dynamic Profile change handler
    const profileTiro = $("#profile-tiro-reto");
    const profileCurvas = $("#profile-curvas");
    if (profileTiro && profileCurvas) {
        profileCurvas.addEventListener("change", () => {
            if (profileCurvas.checked) {
                $('#param-min-area').value = 3;
                $('#param-close-size').value = 3;
                $('#param-open-size').value = 0;
                $('#param-threshold').value = 0.40;
                if ($('#param-calc-entrelinhas')) $('#param-calc-entrelinhas').checked = true;
            }
        });
        profileTiro.addEventListener("change", () => {
            if (profileTiro.checked) {
                $('#param-min-area').value = 10;
                $('#param-close-size').value = 5;
                $('#param-open-size').value = 3;
                $('#param-threshold').value = 0.50;
            }
        });
    }

    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            const step = btn.getAttribute('data-step');
            if (step) showStep(step);
        });
    });

    updateResultLinks(false);
    loadAvailableGpkgs();
}

function setupUpload(zoneSelector, inputSelector, type, labelSelector) {
    const zone = $(zoneSelector);
    const input = $(inputSelector);
    const label = $(labelSelector);
    if (!zone || !input || !label) return;
    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
        if (input.files.length > 0) {
            uploadFile(input.files[0], type, label);
        }
    });
}

function uploadFile(file, type, label) {
    const formData = new FormData();
    formData.append('file', file);
    let endpoint = '/api/upload';
    if (type === 'checkpoint') endpoint = '/api/upload-checkpoint';
    if (type === 'mask') endpoint = '/api/upload-mask';
    if (type !== 'checkpoint' && type !== 'mask') {
        formData.append('type', type);
    }
    fetch(endpoint, { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                label.textContent = `Salvo: ${file.name}`;
                label.style.color = '#33cc33';
            } else {
                label.textContent = `Erro: ${data.error || 'Falha'}`;
                label.style.color = '#ff3333';
            }
        })
        .catch(() => {
            label.textContent = 'Erro ao enviar arquivo';
            label.style.color = '#ff3333';
        });
}

function getPayload() {
    const profileChecked = document.querySelector('input[name="profile"]:checked');
    const profile = profileChecked ? profileChecked.value : 'tiro_reto';

    return {
        gpkg: '63996.gpkg',
        buffer_size: Number($('#param-buffer').value),
        tile_size: Number($('#param-tile-size').value),
        stride: Math.max(1, Math.floor(Number($('#param-tile-size').value) / 2)),
        append: false,
        encoder: 'resnet18',
        epochs: Number($('#param-epochs').value),
        batch_size: Number($('#param-batch-size').value),
        lr: 0.0001,
        threshold: Number($('#param-threshold').value),
        min_area: Number($('#param-min-area').value),
        close_size: Number($('#param-close-size').value),
        open_size: Number($('#param-open-size').value),
        skip_postprocess: false,
        export_as_lines: $('#param-export-lines') ? $('#param-export-lines').checked : false,
        profile: profile,
        calc_entrelinhas: $('#param-calc-entrelinhas') ? $('#param-calc-entrelinhas').checked : false,
        checkpoint: 'best'
    };
}

function startAction() {
    clearExecution();
    if (state.mode === 'train') {
        executeStep('/api/start-train', getPayload(), 'train');
    } else if (state.mode === 'train_infer') {
        runTrainInferSequence();
    } else if (state.mode === 'infer_post') {
        executeStep('/api/start-inference', getPayload(), 'inference');
    } else if (state.mode === 'post_only') {
        executeStep('/api/start-postprocess', getPayload(), 'postprocess');
    }
}

function runTrainInferSequence() {
    updateExecutionStatus('Iniciando pré-processamento...');
    fetch('/api/start-preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getPayload())
    })
        .then(resp => resp.json())
        .then(data => {
            if (data.error) return failExecution(data.error);
            return waitForTask('preprocess');
        })
        .then(result => {
            if (!result) return;
            if (result.status !== 'success') return failExecution(result.error || 'Preprocessamento falhou');
            updateExecutionStatus('Pré-processamento concluído. Iniciando treino...');
            return fetch('/api/start-train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(getPayload()) });
        })
        .then(resp => resp && resp.json())
        .then(data => {
            if (!data) return;
            if (data.error) return failExecution(data.error);
            return waitForTask('train');
        })
        .then(result => {
            if (!result) return;
            if (result.status !== 'success') return failExecution(result.error || 'Treinamento falhou');
            updateExecutionStatus('Treinamento concluído. Iniciando inferência...');
            return fetch('/api/start-inference', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(getPayload()) });
        })
        .then(resp => resp && resp.json())
        .then(data => {
            if (!data) return;
            if (data.error) return failExecution(data.error);
            return waitForTask('inference');
        })
        .then(result => {
            if (!result) return;
            if (result.status !== 'success') return failExecution(result.error || 'Inferência falhou');
            completeExecution('Fluxo train_infer concluído com sucesso.');
        })
        .catch(err => failExecution(err.message || err));
}

function executeStep(endpoint, payload, task) {
    updateExecutionStatus('Enviando comando...');
    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                failExecution(data.error);
                return;
            }
            state.task = task;
            setIndicator(true);
            showStep('step-execute');
            pollTask(task);
        })
        .catch(err => failExecution(err.message || err));
}

function waitForTask(task) {
    return new Promise(resolve => {
        const interval = setInterval(() => {
            fetch('/api/status')
                .then(r => r.json())
                .then(st => {
                    const taskState = st[task];
                    if (!taskState) return;
                    if (taskState.status === 'success' || taskState.status === 'failed') {
                        clearInterval(interval);
                        resolve(taskState);
                    }
                })
                .catch(() => {
                    clearInterval(interval);
                    resolve({ status: 'failed', error: 'Erro de comunicação' });
                });
        }, 1000);
    });
}

function pollTask(task) {
    if (state.statusTimer) clearInterval(state.statusTimer);
    state.statusTimer = setInterval(() => {
        fetch('/api/status')
            .then(res => res.json())
            .then(statusData => {
                const taskState = statusData[task];
                if (!taskState) return;
                updateProgress(taskState.progress || 0);
                updateExecutionStatus(`Status: ${taskState.status}`);
                if (taskState.logs) {
                    renderLogs(taskState.logs);
                }
                if (taskState.status === 'success' || taskState.status === 'failed') {
                    clearInterval(state.statusTimer);
                    setIndicator(false);
                    updateResults(taskState);
                    if (taskState.status === 'success') {
                        completeExecution(`Tarefa ${task} finalizada com sucesso!`);
                    } else {
                        failExecution(taskState.error || 'Erro desconhecido');
                    }
                }
            })
            .catch(() => {
                failExecution('Erro ao consultar status da tarefa');
            });
    }, 1000);
}

function updateProgress(value) {
    const bar = $('#execution-progress');
    if (bar) bar.value = value;
}

function updateExecutionStatus(text) {
    const el = $('#execution-status');
    if (el) el.textContent = text;
}

function renderLogs(lines) {
    const log = $('#execution-log');
    if (!log) return;
    log.innerHTML = lines.map(line => `<div class="log-line">${line}</div>`).join('');
    log.scrollTop = log.scrollHeight;
}

function updateResults(taskState) {
    const resultStatus = $('#result-status');
    const resultSummary = $('#result-summary');
    if (!resultStatus || !resultSummary) return;
    if (taskState.status === 'success') {
        resultStatus.textContent = 'Concluído com sucesso';
        let summary = 'Os arquivos foram salvos em data/outputs e o modelo em models/.';
        if (taskState.results && taskState.results.dominant_angle_deg !== undefined) {
            const angle = taskState.results.dominant_angle_deg.toFixed(1);
            const azimuth = taskState.results.dominant_azimuth_deg.toFixed(1);
            summary += `\n\n🍊 **Sentido das Fileiras do Talhão**: ${angle}° (em relação ao Leste) | **Azimute**: ${azimuth}° (Norte = 0°)`;
        }
        if (taskState.results && taskState.results.num_entrelinhas > 0) {
            summary += `\n🔄 **Entrelinhas Curvas**: ${taskState.results.num_entrelinhas} trechos gerados com sucesso!`;
        }
        resultSummary.innerHTML = summary
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        updateResultLinks(true, taskState.results);
    } else if (taskState.status === 'failed') {
        resultStatus.textContent = 'Erro';
        resultSummary.textContent = taskState.error || 'O fluxo falhou. Verifique o log.';
        updateResultLinks(false);
    }
}

function completeExecution(message) {
    $('#execution-status').textContent = message;
    $('#execution-progress').value = 100;
    showStep('step-result');
}

function failExecution(text) {
    $('#execution-status').textContent = `Falha: ${text}`;
    $('#result-status').textContent = 'Falha';
    $('#result-summary').textContent = text;
    setIndicator(false);
    updateResultLinks(false);
}

function clearExecution() {
    $('#execution-progress').value = 0;
    $('#execution-log').innerHTML = '';
    $('#execution-status').textContent = 'Aguardando execução...';
}

function loadAvailableGpkgs() {
    fetch('/api/list-gpkgs')
        .then(res => res.json())
        .then(data => {
            if (data.gpkgs && data.gpkgs.length > 0) {
                console.log('GPKG disponíveis:', data.gpkgs.join(', '));
            }
        })
        .catch(() => {
            console.warn('Não foi possível listar GPKGs');
        });
}

function updateResultLinks(enabled = false, results = {}) {
    const model = $('#download-model');
    const raster = $('#download-raster');
    const geojson = $('#download-geojson');
    const entrelinhas = $('#download-entrelinhas-geojson');
    if (!model || !raster || !geojson) return;
    
    model.href = enabled ? '/api/download/model/best' : '#';
    raster.href = enabled ? '/api/download/raster' : '#';
    geojson.href = enabled ? '/api/download/vector/geojson' : '#';
    if (entrelinhas) {
        const hasEntrelinhas = enabled && results && (results.num_entrelinhas > 0 || results.entrelinhas_geojson);
        entrelinhas.href = hasEntrelinhas ? '/api/download/vector/entrelinhas_geojson' : '#';
        entrelinhas.classList.toggle('disabled', !hasEntrelinhas);
        entrelinhas.style.display = hasEntrelinhas ? 'inline-block' : 'none';
    }
    
    [model, raster, geojson].forEach(el => el.classList.toggle('disabled', !enabled));
}

window.addEventListener('DOMContentLoaded', initUI);
