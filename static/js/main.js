import { updateSettings } from './settings.js';
import { logMessage } from './log.js';
import {
    updateSlices,
    clearImage,
    loadNiftiImage,
    animate,
    setViewMode,
    updateVolumeThreshold,
    updateVolumeSteps,
    updateVolumeClip,
} from './mriViewer.js';

let isImageLoaded = false;
let currentScanBlob = null;
let currentExampleId = null;
// clearImage() wipes #mri-viewer-container's innerHTML down to just the
// upload hint, losing the example-scan buttons rendered by Jinja on first
// load - snapshot that markup once so it can be restored after a clear.
let originalViewerHTML = null;

function bindExampleButtons() {
    document.querySelectorAll('.example-button').forEach(button => {
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!isImageLoaded) loadExampleFile(button.dataset.exampleId, button.textContent.trim());
        });
    });
}

function toggleTheme() {
    const isDark = document.documentElement.dataset.theme === 'dark';
    if (isDark) {
        delete document.documentElement.dataset.theme;
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.dataset.theme = 'dark';
        localStorage.setItem('theme', 'dark');
    }
}

// File handling functions
function handleFileUpload(event) {
    if (isImageLoaded) return;
    const file = event.target.files ? event.target.files[0] : event.dataTransfer.files[0];
    if (file && file.name.endsWith('.nii')) {
        document.getElementById('mri-viewer-title').textContent = `3D MRI Viewer - ${file.name}`;
        currentScanBlob = file;
        currentExampleId = null;

        const reader = new FileReader();
        reader.onload = function(e) {
            const arrayBuffer = e.target.result;
            const niftiInfo = loadNiftiImage(arrayBuffer);

            console.log('NIfTI header loaded:', niftiInfo);

            isImageLoaded = true;
            updateUploadState();
            animate(); // Start animation after loading
        };
        reader.readAsArrayBuffer(file);
    } else {
        alert('Please upload a valid .nii file');
    }
}

function updateUploadState() {
    const dropZone = document.getElementById('mri-viewer-container');
    const clearButton = document.getElementById('clear-button');
    const classifyButton = document.getElementById('classify-button');
    const classifyOverlay = document.querySelector('.classify-overlay');
    if (isImageLoaded) {
        dropZone.removeEventListener('click', triggerFileInput);
        dropZone.classList.remove('cursor-pointer');
        clearButton.classList.remove('hidden');
        if (classifyButton) {
            classifyButton.disabled = false;
            classifyButton.classList.remove('hidden');
        }
        classifyOverlay?.classList.remove('hidden');
    } else {
        dropZone.addEventListener('click', triggerFileInput);
        dropZone.classList.add('cursor-pointer');
        clearButton.classList.add('hidden');
        if (classifyButton) classifyButton.disabled = true;
        classifyOverlay?.classList.add('hidden');
    }
}

function triggerFileInput() {
    document.getElementById('file-input').click();
}

function loadExampleFile(exampleId, label) {
    // A compressed preview - fast to load and plenty detailed for viewing.
    // Classification (if requested) uses the full-resolution original
    // server-side instead, so accuracy isn't affected by this downsampling.
    fetch(`/examples/${exampleId}/preview`)
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => {
            currentScanBlob = new Blob([arrayBuffer]);
            currentExampleId = exampleId;

            const niftiInfo = loadNiftiImage(arrayBuffer);

            console.log('NIfTI info:', niftiInfo);

            isImageLoaded = true;
            updateUploadState();

            document.getElementById('mri-viewer-title').textContent = `3D MRI Viewer - ${label} (preview)`;
            animate(); // Start animation after loading
        })
        .catch(error => console.error('Error loading example preview:', error));
}

function handleClearImage() {
    clearImage();
    if (originalViewerHTML !== null) {
        document.getElementById('mri-viewer-container').innerHTML = originalViewerHTML;
        bindExampleButtons();
    }
    isImageLoaded = false;
    currentScanBlob = null;
    currentExampleId = null;
    updateUploadState();
    document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer';

    const results = document.getElementById('classify-results');
    if (results) results.innerHTML = '';

    // Reset sliders
    document.getElementById('axial-slider').value = 50;
    document.getElementById('sagittal-slider').value = 50;
    document.getElementById('coronal-slider').value = 50;
}

function renderClassifyResults(data) {
    const container = document.getElementById('classify-results');
    if (!container) return;

    // The button's job is done once a result is showing - leaving it
    // visible just overlaps the detail panel when expanded.
    document.getElementById('classify-button')?.classList.add('hidden');

    const top = data.predictions.find(p => p.label === data.predicted_class) || data.predictions[0];
    const topPct = (top.probability * 100).toFixed(1);

    const bars = data.predictions.map(({ label, probability }) => {
        const pct = (probability * 100).toFixed(1);
        const isTop = label === data.predicted_class;
        return `
            <div class="class-bar${isTop ? ' is-top' : ''}">
                <div class="class-bar-label"><span>${label}</span><span>${pct}%</span></div>
                <div class="class-bar-track"><div class="class-bar-fill" style="width: ${pct}%"></div></div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="result-summary">
            <span class="result-label">${data.predicted_class}</span>
            <span class="result-pct">${topPct}%</span>
            <button type="button" class="result-toggle">More details</button>
        </div>
        <div class="result-detail hidden">
            ${bars}
            <p class="result-caveat">Demo simplification: one 2D slice, no skull-stripping - this confidence score isn't a clinically meaningful result.</p>
        </div>
    `;

    container.querySelector('.result-toggle')?.addEventListener('click', (e) => {
        const detail = container.querySelector('.result-detail');
        detail?.classList.toggle('hidden');
        e.target.textContent = detail?.classList.contains('hidden') ? 'More details' : 'Hide';
    });
}

async function classifyScan() {
    if (!currentScanBlob) return;
    const button = document.getElementById('classify-button');
    button.disabled = true;
    logMessage('Classifying scan…');

    try {
        const formData = new FormData();
        if (currentExampleId) {
            // Classify the full-resolution original server-side, not the
            // compressed preview blob used for the 3D viewer.
            formData.append('example_id', currentExampleId);
        } else {
            formData.append('file', currentScanBlob, 'scan.nii');
        }
        const response = await fetch('/classify', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok) {
            logMessage(`Classification failed: ${data.error || response.statusText}`, 'error');
            return;
        }

        renderClassifyResults(data);
        logMessage(`Classified as ${data.predicted_class}`, 'success');
    } catch (error) {
        logMessage(`Classification failed: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
    }
}

async function runPipelineAction(button) {
    const action = button.dataset.action;
    const label = button.textContent.trim();
    button.disabled = true;
    logMessage(`Running ${label}…`);

    try {
        const response = await fetch('/run', {
            method: 'POST',
            body: new URLSearchParams({ action }),
        });
        const text = await response.text();
        logMessage(text, response.ok ? 'success' : 'error');
    } catch (error) {
        logMessage(`${label} failed: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
    }
}

window.onload = function() {
    // The 3D scene/renderer is only initialized once a scan is actually
    // loaded (loadNiftiImage calls initMRIViewer itself) - initializing it
    // eagerly here would wipe the upload hint/example-load button before
    // anyone can see or click them.
    if (!document.getElementById('mri-viewer-container')) {
        console.error('MRI viewer container not found');
    }

    const dropZone = document.getElementById('mri-viewer-container');
    const fileInput = document.getElementById('file-input');
    const clearButton = document.getElementById('clear-button');

    updateUploadState();

    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
    document.getElementById('update-settings')?.addEventListener('click', updateSettings);
    document.getElementById('classify-button')?.addEventListener('click', classifyScan);
    originalViewerHTML = document.getElementById('mri-viewer-container').innerHTML;
    bindExampleButtons();

    document.querySelectorAll('.pipeline-action').forEach(button => {
        button.addEventListener('click', () => runPipelineAction(button));
    });

    document.querySelectorAll('.mode-btn[data-mode]').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn[data-mode]').forEach(b => b.classList.remove('active'));
            button.classList.add('active');
            setViewMode(button.dataset.mode);
        });
    });

    document.querySelectorAll('.threshold-btn').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.threshold-btn').forEach(b => b.classList.remove('active'));
            button.classList.add('active');
            updateVolumeThreshold(button.dataset.threshold);
        });
    });

    document.querySelectorAll('details.tool').forEach(tool => {
        tool.addEventListener('toggle', () => {
            if (!tool.open) return;
            document.querySelectorAll('details.tool').forEach(other => {
                if (other !== tool) other.open = false;
            });

            // Right-anchored by default; if that pushes it past the left
            // edge of the viewport (common in a narrow embed), re-anchor
            // from the left instead so it stays fully visible.
            const panel = tool.querySelector('.tool-panel');
            if (!panel) return;
            panel.style.right = '';
            panel.style.left = '';
            requestAnimationFrame(() => {
                const rect = panel.getBoundingClientRect();
                if (rect.left < 8) {
                    const toolRect = tool.getBoundingClientRect();
                    panel.style.right = 'auto';
                    panel.style.left = `${8 - toolRect.left}px`;
                }
            });
        });
    });

    document.getElementById('volume-steps')?.addEventListener('input', function() {
        updateVolumeSteps(this.value);
    });

    if (dropZone) {
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (!isImageLoaded) {
                this.classList.add('drag-over');
            }
        });

        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('drag-over');
            if (!isImageLoaded) {
                handleFileUpload(e);
            }
        });
    } else {
        console.error('Drop zone element not found');
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    } else {
        console.error('File input element not found');
    }

    if (clearButton) {
        clearButton.addEventListener('click', handleClearImage);
    } else {
        console.error('Clear button element not found');
    }

    // Add event listeners for sliders. Each also feeds the volume view's
    // clip planes (no-op in slices mode) - moving a slider away from its
    // default reveals a cutaway through the volume at that position,
    // matching the cross-section the same slider shows in slices mode.
    document.getElementById('axial-slider').addEventListener('input', function() {
        updateSlices(parseInt(this.value), parseInt(document.getElementById('sagittal-slider').value), parseInt(document.getElementById('coronal-slider').value));
        updateVolumeClip('axial', this.value / 100);
    });
    document.getElementById('sagittal-slider').addEventListener('input', function() {
        updateSlices(parseInt(document.getElementById('axial-slider').value), parseInt(this.value), parseInt(document.getElementById('coronal-slider').value));
        updateVolumeClip('sagittal', this.value / 100);
    });
    document.getElementById('coronal-slider').addEventListener('input', function() {
        updateSlices(parseInt(document.getElementById('axial-slider').value), parseInt(document.getElementById('sagittal-slider').value), parseInt(this.value));
        updateVolumeClip('coronal', this.value / 100);
    });
};
