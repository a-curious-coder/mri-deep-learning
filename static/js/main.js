import { setInitialTheme } from './theme.js';
import { updateSettings } from './settings.js';
import { updateSlices, clearImage, loadNiftiImage, animate } from './mriViewer.js';
import { initMRIViewer, updateCanvasSize } from './mriViewer.js';

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
    updateUploadState();
    document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer';
    
    // Reset sliders
    document.getElementById('axial-slider').value = 50;
    document.getElementById('sagittal-slider').value = 50;
    document.getElementById('coronal-slider').value = 50;
}

function applyCanvasSize() {
    const width = parseInt(document.getElementById('canvas-width').value);
    const height = parseInt(document.getElementById('canvas-height').value);
    updateCanvasSize(width, height);
}

window.onload = function() {
    setInitialTheme();
    
    // Initialize MRI viewer
    const mriViewerContainer = document.getElementById('mri-viewer-container');
    if (mriViewerContainer) {
        initMRIViewer();
    } else {
        console.error('MRI viewer container not found');
    }

    // Change this line
    const dropZone = document.getElementById('mri-viewer-container');
    const fileInput = document.getElementById('file-input');
    const clearButton = document.getElementById('clear-button');

    if (dropZone) {
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

// Export functions for use in HTML
window.handleFileUpload = handleFileUpload;
window.handleClearImage = handleClearImage;
window.updateSettings = updateSettings;
window.applyCanvasSize = applyCanvasSize;  // Add this line

// Remove these lines from the end of the file
// initMRIViewer();
// animate();