import { updateSettings } from './settings.js';
import { logMessage } from './log.js';
import {
    updateSlices,
    clearImage,
    loadNiftiImage,
    animate,
    initMRIViewer,
    updateCanvasSize,
    setViewMode,
    updateVolumeThreshold,
    updateVolumeSteps,
} from './mriViewer.js';

let isImageLoaded = false;
let currentScanBlob = null;

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
    if (isImageLoaded) {
        dropZone.removeEventListener('click', triggerFileInput);
        dropZone.classList.remove('cursor-pointer');
        clearButton.classList.remove('hidden');
        if (classifyButton) classifyButton.disabled = false;
    } else {
        dropZone.addEventListener('click', triggerFileInput);
        dropZone.classList.add('cursor-pointer');
        clearButton.classList.add('hidden');
        if (classifyButton) classifyButton.disabled = true;
    }
}

function triggerFileInput() {
    document.getElementById('file-input').click();
}

// Check if the example.nii file exists
fetch('/static/data/raw/example.nii')
    .then(response => {
        if (response.ok) {
            console.log('example.nii file exists');
            // Load the file automatically
            loadExampleFile();
        } else {
            console.log('example.nii file does not exist');
            updateUploadState();
        }
    })
    .catch(error => {
        console.error('Error checking for example.nii:', error);
        updateUploadState();
    });

function loadExampleFile() {
    fetch('/static/data/raw/example.nii')
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => {
            currentScanBlob = new Blob([arrayBuffer]);

            const niftiInfo = loadNiftiImage(arrayBuffer);

            console.log('NIfTI info:', niftiInfo);

            isImageLoaded = true;
            updateUploadState();

            document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer - example.nii';
            animate(); // Start animation after loading
        })
        .catch(error => console.error('Error loading example.nii:', error));
}

function handleClearImage() {
    clearImage();
    isImageLoaded = false;
    currentScanBlob = null;
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

    container.innerHTML = '';
    data.predictions.forEach(({ label, probability }) => {
        const pct = (probability * 100).toFixed(1);
        const bar = document.createElement('div');
        bar.className = `class-bar${label === data.predicted_class ? ' is-top' : ''}`;
        bar.innerHTML = `
            <div class="class-bar-label"><span>${label}</span><span>${pct}%</span></div>
            <div class="class-bar-track"><div class="class-bar-fill" style="width: ${pct}%"></div></div>
        `;
        container.appendChild(bar);
    });
}

async function classifyScan() {
    if (!currentScanBlob) return;
    const button = document.getElementById('classify-button');
    button.disabled = true;
    logMessage('Classifying scan…');

    try {
        const formData = new FormData();
        formData.append('file', currentScanBlob, 'scan.nii');
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

function applyCanvasSize() {
    const width = parseInt(document.getElementById('canvas-width').value, 10);
    const height = parseInt(document.getElementById('canvas-height').value, 10);
    updateCanvasSize(width, height);
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
    // Initialize MRI viewer
    const mriViewerContainer = document.getElementById('mri-viewer-container');
    if (mriViewerContainer) {
        initMRIViewer();
    } else {
        console.error('MRI viewer container not found');
    }

    const dropZone = document.getElementById('mri-viewer-container');
    const fileInput = document.getElementById('file-input');
    const clearButton = document.getElementById('clear-button');

    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
    document.getElementById('update-settings')?.addEventListener('click', updateSettings);
    document.getElementById('classify-button')?.addEventListener('click', classifyScan);

    document.querySelectorAll('.pipeline-action').forEach(button => {
        button.addEventListener('click', () => runPipelineAction(button));
    });

    document.querySelectorAll('.mode-btn').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            button.classList.add('active');
            setViewMode(button.dataset.mode);
        });
    });

    document.querySelectorAll('details.tool').forEach(tool => {
        tool.addEventListener('toggle', () => {
            if (!tool.open) return;
            document.querySelectorAll('details.tool').forEach(other => {
                if (other !== tool) other.open = false;
            });
        });
    });

    document.getElementById('volume-threshold')?.addEventListener('input', function() {
        updateVolumeThreshold(this.value);
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

    // Add event listeners for sliders
    document.getElementById('axial-slider').addEventListener('input', function() {
        updateSlices(parseInt(this.value), parseInt(document.getElementById('sagittal-slider').value), parseInt(document.getElementById('coronal-slider').value));
    });
    document.getElementById('sagittal-slider').addEventListener('input', function() {
        updateSlices(parseInt(document.getElementById('axial-slider').value), parseInt(this.value), parseInt(document.getElementById('coronal-slider').value));
    });
    document.getElementById('coronal-slider').addEventListener('input', function() {
        updateSlices(parseInt(document.getElementById('axial-slider').value), parseInt(document.getElementById('sagittal-slider').value), parseInt(this.value));
    });

    // Add event listener for the Apply Canvas Size button
    const applyCanvasSizeButton = document.getElementById('apply-canvas-size');
    if (applyCanvasSizeButton) {
        applyCanvasSizeButton.addEventListener('click', applyCanvasSize);
    } else {
        console.error('Apply Canvas Size button not found');
    }
};
