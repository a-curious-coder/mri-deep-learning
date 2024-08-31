let isImageLoaded = false;

function handleFileUpload(event) {
    if (isImageLoaded) return; // Prevent upload if image is already loaded
    const file = event.target.files ? event.target.files[0] : event.dataTransfer.files[0];
    if (file && file.name.endsWith('.nii')) {
        document.getElementById('mri-viewer-title').textContent = `3D MRI Viewer - ${file.name}`;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const arrayBuffer = e.target.result;
            niftiHeader = nifti.readHeader(arrayBuffer);
            niftiImage = nifti.readImage(niftiHeader, arrayBuffer);
            
            console.log('NIfTI header loaded:', niftiHeader);
            console.log('NIfTI image loaded:', niftiImage);
            console.log('Image dimensions:', niftiHeader.dims.slice(1, 4));
            console.log('Data type:', niftiHeader.datatypeCode);
            console.log('Voxel dimensions:', niftiHeader.pixDims.slice(1, 4));
            
            // Initialize the viewer with the middle slices
            const midAxial = Math.floor(niftiHeader.dims[3] / 2);
            const midSagittal = Math.floor(niftiHeader.dims[1] / 2);
            const midCoronal = Math.floor(niftiHeader.dims[2] / 2);
            
            // Reinitialize the MRI viewer
            initMRIViewer();
            
            updateSlices(
                (midAxial / niftiHeader.dims[3]) * 100,
                (midSagittal / niftiHeader.dims[1]) * 100,
                (midCoronal / niftiHeader.dims[2]) * 100
            );
            
            // Update slider ranges
            document.getElementById('axial-slider').max = niftiHeader.dims[3] - 1;
            document.getElementById('sagittal-slider').max = niftiHeader.dims[1] - 1;
            document.getElementById('coronal-slider').max = niftiHeader.dims[2] - 1;
            
            isImageLoaded = true;
            updateUploadState();
        };
        reader.readAsArrayBuffer(file);
    } else {
        alert('Please upload a valid .nii file');
    }
}

function clearImage() {
    // Clear the MRI viewer
    document.getElementById('mri-viewer').innerHTML = '<p class="text-center text-light-green-800 dark:text-gray-300">Drag and drop a .nii file here<br>or click to select</p>';
    isImageLoaded = false;
    updateUploadState();
    
    // Reset the viewer title
    document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer';
    
    // Clear the niftiImage and niftiHeader
    niftiImage = null;
    niftiHeader = null;
    
    // Reset the sliders
    document.getElementById('axial-slider').value = 50;
    document.getElementById('sagittal-slider').value = 50;
    document.getElementById('coronal-slider').value = 50;
    
    // Reset the Three.js scene
    if (scene) {
        while(scene.children.length > 0){ 
            scene.remove(scene.children[0]); 
        }
    }
    if (renderer) {
        renderer.dispose();
    }
    scene = null;
    camera = null;
    renderer = null;
    controls = null;
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

function loadExampleFile() {
    fetch('/static/data/raw/example.nii')
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => {
            niftiHeader = nifti.readHeader(arrayBuffer);
            niftiImage = nifti.readImage(niftiHeader, arrayBuffer);
            
            console.log('NIfTI header:', niftiHeader);
            console.log('NIfTI image data:', niftiImage);
            console.log('Image dimensions:', niftiHeader.dims.slice(1, 4));
            console.log('Data type:', niftiHeader.datatypeCode);
            console.log('Voxel dimensions:', niftiHeader.pixDims.slice(1, 4));

            updateSlices(50, 50, 50);
            
            // Set isImageLoaded to true and update the upload state
            isImageLoaded = true;
            updateUploadState();
            
            // Update the viewer title
            document.getElementById('mri-viewer-title').textContent = '3D MRI Viewer - example.nii';
        })
        .catch(error => console.error('Error loading example.nii:', error));
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
