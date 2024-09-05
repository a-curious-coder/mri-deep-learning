let scene, camera, renderer, niftiImage, niftiHeader;
let axialPlane, sagittalPlane, coronalPlane;
let axialLine, sagittalLine, coronalLine;
let controls;
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
let isNiftiLoaded = false;

export function initMRIViewer() {
    // Get the container element
    const container = document.getElementById('mri-viewer-container');
    
    // Check if the container element exists
    if (!container) {
        console.error('MRI viewer container not found');
        return;
    }

    // Get the container dimensions
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Scene setup
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, containerWidth / containerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(containerWidth, containerHeight);

    // Clear and append renderer to DOM
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Camera positioning
    camera.position.set(1.5, 1.5, 1.5);
    camera.lookAt(0, 0, 0);

    // Controls setup
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.enableZoom = true;

    // Add container boundaries
    const containerGeometry = new THREE.BoxGeometry(1, 1, 1);
    const edges = new THREE.EdgesGeometry(containerGeometry);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff }));
    scene.add(line);

    // Create planes
    const planeGeometry = new THREE.PlaneGeometry(1, 1);
    const planeMaterial = new THREE.MeshBasicMaterial({ 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.8 
    });
    
    [axialPlane, sagittalPlane, coronalPlane] = ['axial', 'sagittal', 'coronal'].map(
        type => new THREE.Mesh(planeGeometry, planeMaterial.clone())
    );

    // Set plane rotations
    axialPlane.rotation.x = Math.PI / 2;
    sagittalPlane.rotation.y = Math.PI / 2;

    // Add planes to scene
    scene.add(axialPlane, sagittalPlane, coronalPlane);

    // Create lines
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.5, 0, 0),
        new THREE.Vector3(0.5, 0, 0)
    ]);

    const lineColors = [0xff0000, 0x00ff00, 0x0000ff];
    [axialLine, sagittalLine, coronalLine] = lineColors.map(
        color => new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color }))
    );

    // Set line rotations
    axialLine.rotation.x = Math.PI / 2;
    sagittalLine.rotation.y = Math.PI / 2;

    // Add lines to scene
    scene.add(axialLine, sagittalLine, coronalLine);

    animate();
}

export function animate() {
    if (isNiftiLoaded && renderer && scene && camera && controls) {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
}

export function updateSlices(axialValue, sagittalValue, coronalValue) {
    console.log('Updating slices:', { axialValue, sagittalValue, coronalValue });

    if (!niftiImage || !axialPlane || !sagittalPlane || !coronalPlane) {
        console.warn('Required objects not initialized. Aborting slice update.');
        return;
    }

    const dims = niftiHeader.dims.slice(1, 4);
    console.log('Image dimensions:', dims);

    const aspectRatio = {
        x: dims[0] / Math.max(...dims),
        y: dims[1] / Math.max(...dims),
        z: dims[2] / Math.max(...dims)
    };
    console.log('Aspect ratio:', aspectRatio);

    updatePlanePositions(axialValue, sagittalValue, coronalValue, aspectRatio, dims);

    console.log('Updating axial plane texture');
    updatePlaneTexture(axialPlane, axialValue, 'axial');
    console.log('Updating sagittal plane texture');
    updatePlaneTexture(sagittalPlane, sagittalValue, 'sagittal');
    console.log('Updating coronal plane texture');
    updatePlaneTexture(coronalPlane, coronalValue, 'coronal');

    console.log('Slice update complete');
}

function updatePlanePositions(axialIndex, sagittalIndex, coronalIndex, aspectRatio, dims) {
    axialPlane.position.y = ((axialIndex / (dims[2] - 1)) - 0.5) * aspectRatio.z;
    sagittalPlane.position.x = ((sagittalIndex / (dims[0] - 1)) - 0.5) * aspectRatio.x;
    coronalPlane.position.z = ((coronalIndex / (dims[1] - 1)) - 0.5) * aspectRatio.y;

    axialLine.position.y = axialPlane.position.y;
    sagittalLine.position.x = sagittalPlane.position.x;
    coronalLine.position.z = coronalPlane.position.z;
}

function updatePlaneTexture(plane, sliceIndex, orientation) {
    const sliceData = extractSliceData(sliceIndex, orientation);
    const normalizedData = normalizeSliceData(sliceData);
    const texture = createTextureFromData(normalizedData, sliceData.width, sliceData.height);
    updatePlaneMaterial(plane, texture);
    adjustPlaneScale(plane, sliceData.width, sliceData.height, orientation);
}

function extractSliceData(sliceIndex, orientation) {
    const dims = niftiHeader.dims;
    const imageData = new Uint16Array(niftiImage);
    let width, height, sliceData;

    switch (orientation) {
        case 'axial':
            width = dims[1];
            height = dims[2];
            sliceData = extractAxialSlice(imageData, dims, sliceIndex, width, height);
            break;
        case 'sagittal':
            width = dims[2];
            height = dims[3];
            sliceData = extractSagittalSlice(imageData, dims, sliceIndex, width, height);
            break;
        case 'coronal':
            width = dims[1];
            height = dims[3];
            sliceData = extractCoronalSlice(imageData, dims, sliceIndex, width, height);
            break;
    }

    return { data: sliceData, width, height };
}

function extractAxialSlice(imageData, dims, sliceIndex, width, height) {
    const sliceData = new Uint16Array(width * height);
    const z = sliceIndex;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            sliceData[y * width + x] = imageData[z * width * height + y * width + x];
        }
    }
    return sliceData;
}

function extractSagittalSlice(imageData, dims, sliceIndex, width, height) {
    let sliceData = new Uint16Array(width * height);
    const x = sliceIndex;
    for (let z = 0; z < height; z++) {
        for (let y = 0; y < width; y++) {
            sliceData[(height - 1 - z) * width + y] = imageData[(height - 1 - z) * dims[1] * dims[2] + y * dims[1] + x];
        }
    }
    let mirroredSliceData = new Uint16Array(width * height);
    for (let z = 0; z < height; z++) {
        for (let y = 0; y < width; y++) {
            mirroredSliceData[z * width + (width - 1 - y)] = sliceData[z * width + y];
        }
    }
    sliceData = mirroredSliceData;
    return sliceData;
}

function extractCoronalSlice(imageData, dims, sliceIndex, width, height) {
    const sliceData = new Uint16Array(width * height);
    const y = sliceIndex;
    for (let z = 0; z < height; z++) {
        for (let x = 0; x < width; x++) {
            sliceData[z * width + x] = imageData[z * dims[1] * dims[2] + y * dims[1] + x];
        }
    }
    return sliceData;
}

function normalizeSliceData(sliceData) {
    const min = Math.min(...sliceData.data);
    const max = Math.max(...sliceData.data);
    const range = max - min;

    const normalizedData = new Uint8Array(sliceData.width * sliceData.height * 4);
    for (let i = 0; i < sliceData.data.length; i++) {
        const normalizedValue = Math.round(((sliceData.data[i] - min) / range) * 255);
        normalizedData[i * 4] = normalizedValue;     // Red channel
        normalizedData[i * 4 + 1] = normalizedValue; // Green channel
        normalizedData[i * 4 + 2] = normalizedValue; // Blue channel
        normalizedData[i * 4 + 3] = 255;             // Alpha channel (fully opaque)
    }

    return normalizedData;
}

function createTextureFromData(normalizedData, width, height) {
    const texture = new THREE.DataTexture(normalizedData, width, height, THREE.RGBAFormat);
    texture.needsUpdate = true;
    return texture;
}

function updatePlaneMaterial(plane, texture) {
    plane.material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 1  // Increase opacity to make slices fully visible
    });
}

function adjustPlaneScale(plane, width, height, orientation) {
    const dims = niftiHeader.dims.slice(1, 4);
    const maxDim = Math.max(...dims);
    
    switch (orientation) {
        case 'axial':
            plane.scale.set(dims[0] / maxDim, dims[1] / maxDim, 1);
            break;
        case 'sagittal':
            plane.scale.set(dims[1] / maxDim, dims[2] / maxDim, 1);
            break;
        case 'coronal':
            plane.scale.set(dims[0] / maxDim, dims[2] / maxDim, 1);
            break;
    }
}

export function clearImage() {
    const viewerContainer = document.getElementById('mri-viewer-container');
    viewerContainer.innerHTML = '<p class="text-center text-light-green-800 dark:text-gray-300">Drag and drop a .nii file here<br>or click to select</p>';
    
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
    niftiImage = null;
    niftiHeader = null;
    isNiftiLoaded = false;
}

export function updateCanvasSize(width, height) {
    if (!renderer || !camera) return;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
}

export function loadNiftiImage(arrayBuffer) {
    niftiHeader = nifti.readHeader(arrayBuffer);
    niftiImage = nifti.readImage(niftiHeader, arrayBuffer);
    
    console.log('NIfTI header:', niftiHeader);
    console.log('Image dimensions:', niftiHeader.dims.slice(1, 4));
    console.log('Data type:', niftiHeader.datatypeCode);
    console.log('Voxel dimensions:', niftiHeader.pixDims.slice(1, 4));

    isNiftiLoaded = true;
    initMRIViewer();

    const dims = niftiHeader.dims.slice(1, 4);
    const midAxial = Math.floor(dims[2] / 2);
    const midSagittal = Math.floor(dims[0] / 2);
    const midCoronal = Math.floor(dims[1] / 2);

    // Update slider ranges to match image dimensions
    const axialSlider = document.getElementById('axial-slider');
    const sagittalSlider = document.getElementById('sagittal-slider');
    const coronalSlider = document.getElementById('coronal-slider');

    axialSlider.min = 0;
    axialSlider.max = dims[2] - 1;
    axialSlider.value = midAxial;

    sagittalSlider.min = 0;
    sagittalSlider.max = dims[0] - 1;
    sagittalSlider.value = midSagittal;

    coronalSlider.min = 0;
    coronalSlider.max = dims[1] - 1;
    coronalSlider.value = midCoronal;

    // Initialize the slices with the middle values
    updateSlices(midAxial, midSagittal, midCoronal);

    return {
        dimensions: dims,
        dataType: niftiHeader.datatypeCode,
        voxelDimensions: niftiHeader.pixDims.slice(1, 4)
    };
}
