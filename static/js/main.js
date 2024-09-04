import { setInitialTheme } from './theme.js';
import { updateSettings } from './settings.js';
import { updateSlices, clearImage, loadNiftiImage } from './mriViewer.js';
import { initMRIViewer, updateCanvasSize, animate } from './mriViewer.js';

let isImageLoaded = false;
let containerWidth = 800;
let containerHeight = 500;

// File handling functions
function handleFileUpload(event) {
    if (isImageLoaded) return;
    const file = event.target.files ? event.target.files[0] : event.dataTransfer.files[0];
    if (file && file.name.endsWith('.nii')) {
        document.getElementById('mri-viewer-title').textContent = `3D MRI Viewer - ${file.name}`;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const arrayBuffer = e.target.result;
            const niftiInfo = loadNiftiImage(arrayBuffer);
            
            console.log('NIfTI header loaded:', niftiInfo);
            
            isImageLoaded = true;
            updateUploadState();
        };
        reader.readAsArrayBuffer(file);
    } else {
        alert('Please upload a valid .nii file');
    }
}

function updateUploadState() {
    const dropZone = document.getElementById('mri-viewer');
    const clearButton = document.getElementById('clear-button');
    if (isImageLoaded) {
        dropZone.removeEventListener('click', triggerFileInput);
        dropZone.classList.remove('cursor-pointer');
        clearButton.classList.remove('hidden');
    } else {
        dropZone.addEventListener('click', triggerFileInput);
        dropZone.classList.add('cursor-pointer');
        clearButton.classList.add('hidden');
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
            const niftiInfo = loadNiftiImage(arrayBuffer);
            
            console.log('NIfTI info:', niftiInfo);

            // Set isImageLoaded to true and update the upload state
            isImageLoaded = true;
            updateUploadState();
            
            // Update the viewer title
            document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer - example.nii';
        })
        .catch(error => console.error('Error loading example.nii:', error));
}

function handleClearImage() {
    clearImage();
    isImageLoaded = false;
    updateUploadState();
    document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer';
    
    // Reset sliders
    document.getElementById('axial-slider').value = 50;
    document.getElementById('sagittal-slider').value = 50;
    document.getElementById('coronal-slider').value = 50;
}

document.addEventListener('DOMContentLoaded', function() {
    setInitialTheme();
    
    // Initialize MRI viewer
    initMRIViewer();

    const dropZone = document.getElementById('mri-viewer');
    const fileInput = document.getElementById('file-input');
    const clearButton = document.getElementById('clear-button');

    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!isImageLoaded) {
            this.classList.add('bg-light-green-300', 'dark:bg-gray-600');
        }
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('bg-light-green-300', 'dark:bg-gray-600');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('bg-light-green-300', 'dark:bg-gray-600');
        if (!isImageLoaded) {
            handleFileUpload(e);
        }
    });

    fileInput.addEventListener('change', handleFileUpload);

    clearButton.addEventListener('click', handleClearImage);

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
    document.querySelector('button[onclick="applyCanvasSize()"]').addEventListener('click', applyCanvasSize);
});

// Export functions for use in HTML
window.handleFileUpload = handleFileUpload;
window.handleClearImage = handleClearImage;
window.updateSettings = updateSettings;

initMRIViewer();
animate();

// Update the applyCanvasSize function
function applyCanvasSize() {
    const width = parseInt(document.getElementById('canvas-width').value);
    const height = parseInt(document.getElementById('canvas-height').value);
    
    // Update the canvas size
    updateCanvasSize(width, height);
}

// Make the function globally accessible
window.applyCanvasSize = applyCanvasSize;