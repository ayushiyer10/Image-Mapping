/**
 * Single-Image Surface Map Estimation & Real-Time Physical Relighting
 * Frontend Application JavaScript
 */

const state = {
    mode: 'benchmark', // 'benchmark' or 'upload'
    method: 'both',    // 'both', 'unet', or 'dip'
    activeNormalView: 'unet', // 'unet' or 'dip'
    samples: [],
    activeAssetId: null,
    currentMapData: null,
    showingGT: false,
    lightPos: { x: 0.0, y: 0.0, z: 1.5 },
    isDraggingLight: false
};

let scene, camera, renderer, planeMesh, shaderMaterial;
let defaultTextures = {};

const elements = {
    btnBenchmarkMode: document.getElementById('btn-benchmark-mode'),
    btnUploadMode: document.getElementById('btn-upload-mode'),
    benchmarkControls: document.getElementById('benchmark-controls'),
    uploadControls: document.getElementById('upload-controls'),
    sampleSelect: document.getElementById('sample-select'),
    btnRunBenchmark: document.getElementById('btn-run-benchmark'),
    fileInput: document.getElementById('file-input'),
    btnWebcam: document.getElementById('btn-webcam'),

    // Method selection pills
    methodBoth: document.getElementById('method-both'),
    methodUnet: document.getElementById('method-unet'),
    methodDip: document.getElementById('method-dip'),

    // Scorecard Metrics
    metricMae: document.getElementById('metric-mae'),
    metricMaeSub: document.getElementById('metric-mae-sub'),
    metricPct1125: document.getElementById('metric-pct-11-25'),
    metricPctSub: document.getElementById('metric-pct-sub'),
    metricPct225: document.getElementById('metric-pct-22-5'),
    metricNormalPsnr: document.getElementById('metric-normal-psnr'),
    metricRoughSsim: document.getElementById('metric-rough-ssim'),
    metricHeightSsim: document.getElementById('metric-height-ssim'),
    gtStatusBadge: document.getElementById('gt-status-badge'),
    
    // 2D Previews & View Selectors
    imgInputRgb: document.getElementById('img-input-rgb'),
    imgNormalMap: document.getElementById('img-normal-map'),
    imgRoughnessMap: document.getElementById('img-roughness-map'),
    imgHeightMap: document.getElementById('img-height-map'),
    toggleGtBtn: document.getElementById('toggle-gt-btn'),
    gtToggleLabel: document.getElementById('gt-toggle-label'),
    badgeSourceType: document.getElementById('badge-source-type'),

    viewNormalUnet: document.getElementById('view-normal-unet'),
    viewNormalDip: document.getElementById('view-normal-dip'),
    badgeNormalTech: document.getElementById('badge-normal-tech'),

    titleNormal: document.getElementById('title-normal'),
    titleRoughness: document.getElementById('title-roughness'),
    titleHeight: document.getElementById('title-height'),

    // 3D Viewport
    canvasContainer: document.getElementById('canvas-container'),
    btnResetLight: document.getElementById('btn-reset-light'),
    lightPosReadout: document.getElementById('light-pos-readout'),
    
    sliderLightZ: document.getElementById('slider-light-z'),
    sliderLightIntensity: document.getElementById('slider-light-intensity'),
    sliderSpecular: document.getElementById('slider-specular'),
    sliderDisplacement: document.getElementById('slider-displacement'),
    colorLight: document.getElementById('color-light'),
    
    valLightZ: document.getElementById('val-light-z'),
    valLightIntensity: document.getElementById('val-light-intensity'),
    valSpecular: document.getElementById('val-specular'),
    valDisplacement: document.getElementById('val-displacement')
};

document.addEventListener('DOMContentLoaded', async () => {
    initModeSwitcher();
    initMethodSelector();
    initNormalViewToggle();
    initSliderControls();
    await initThreeJS();
    await fetchSamples();
});

function initModeSwitcher() {
    elements.btnBenchmarkMode.addEventListener('click', () => setMode('benchmark'));
    elements.btnUploadMode.addEventListener('click', () => setMode('upload'));

    elements.sampleSelect.addEventListener('change', (e) => {
        state.activeAssetId = e.target.value;
    });

    elements.btnRunBenchmark.addEventListener('click', () => {
        if (state.activeAssetId) {
            runMapGeneration({ assetId: state.activeAssetId });
        }
    });

    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            runMapGeneration({ file: e.target.files[0] });
        }
    });

    elements.btnWebcam.addEventListener('click', () => {
        captureWebcamPhoto();
    });

    elements.toggleGtBtn.addEventListener('click', () => {
        state.showingGT = !state.showingGT;
        update2DMapPreviews();
    });
}

function initMethodSelector() {
    elements.methodBoth.addEventListener('click', () => setMethod('both'));
    elements.methodUnet.addEventListener('click', () => setMethod('unet'));
    elements.methodDip.addEventListener('click', () => setMethod('dip'));
}

function setMethod(method) {
    state.method = method;
    const buttons = [
        { el: elements.methodBoth, id: 'both' },
        { el: elements.methodUnet, id: 'unet' },
        { el: elements.methodDip, id: 'dip' }
    ];

    buttons.forEach(b => {
        if (b.id === method) {
            b.el.className = "px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 text-white font-medium shadow-md";
        } else {
            b.el.className = "px-3 py-1.5 rounded-lg text-gray-400 hover:text-white font-medium";
        }
    });

    if (state.currentMapData) {
        update2DMapPreviews();
        updateMetricsScorecard(state.currentMapData.metrics);
    }
}

function initNormalViewToggle() {
    elements.viewNormalUnet.addEventListener('click', () => {
        state.activeNormalView = 'unet';
        elements.viewNormalUnet.className = "px-2 py-0.5 rounded bg-indigo-600 text-white font-medium";
        elements.viewNormalDip.className = "px-2 py-0.5 rounded text-gray-400 hover:text-white font-medium";
        update2DMapPreviews();
    });

    elements.viewNormalDip.addEventListener('click', () => {
        state.activeNormalView = 'dip';
        elements.viewNormalDip.className = "px-2 py-0.5 rounded bg-indigo-600 text-white font-medium";
        elements.viewNormalUnet.className = "px-2 py-0.5 rounded text-gray-400 hover:text-white font-medium";
        update2DMapPreviews();
    });
}

function setMode(mode) {
    state.mode = mode;
    if (mode === 'benchmark') {
        elements.btnBenchmarkMode.className = "flex-1 md:flex-none px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2";
        elements.btnUploadMode.className = "flex-1 md:flex-none px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 text-gray-400 hover:text-white flex items-center justify-center space-x-2";
        elements.benchmarkControls.classList.remove('hidden');
        elements.uploadControls.classList.add('hidden');
    } else {
        elements.btnUploadMode.className = "flex-1 md:flex-none px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2";
        elements.btnBenchmarkMode.className = "flex-1 md:flex-none px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 text-gray-400 hover:text-white flex items-center justify-center space-x-2";
        elements.uploadControls.classList.remove('hidden');
        elements.benchmarkControls.classList.add('hidden');
    }
}

async function fetchSamples() {
    try {
        const resp = await fetch('/api/samples');
        const data = await resp.json();
        state.samples = data.samples || [];
        populateSampleDropdown();

        if (state.samples.length > 0) {
            state.activeAssetId = state.samples[0].asset_id;
            elements.sampleSelect.value = state.activeAssetId;
            runMapGeneration({ assetId: state.activeAssetId });
        }
    } catch (err) {
        console.error('Failed to load dataset samples:', err);
    }
}

function populateSampleDropdown() {
    elements.sampleSelect.innerHTML = '';
    
    if (state.samples.length === 0) {
        elements.sampleSelect.innerHTML = '<option value="">No samples found</option>';
        return;
    }

    state.samples.forEach(sample => {
        const opt = document.createElement('option');
        opt.value = sample.asset_id;
        opt.textContent = `${sample.name} ${sample.has_gt ? '✓ (GT)' : ''}`;
        elements.sampleSelect.appendChild(opt);
    });

    state.activeAssetId = state.samples[0].asset_id;
    elements.sampleSelect.value = state.activeAssetId;
}

async function runMapGeneration({ assetId = null, file = null }) {
    elements.btnRunBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1"></i> Estimating...';
    elements.btnRunBenchmark.disabled = true;

    try {
        const formData = new FormData();
        formData.append('method', state.method);

        if (assetId) {
            formData.append('asset_id', assetId);
        } else if (file) {
            formData.append('file', file);
        }

        const resp = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const errorJson = await resp.json();
            alert(`Error: ${errorJson.detail || 'Failed to process image'}`);
            return;
        }

        const data = await resp.json();
        state.currentMapData = data;
        state.showingGT = false;

        update2DMapPreviews();
        updateMetricsScorecard(data.metrics);
        update3DTextures(data.maps);

    } catch (err) {
        console.error('Surface Map estimation failed:', err);
    } finally {
        elements.btnRunBenchmark.innerHTML = '<i class="fa-solid fa-play mr-1"></i> Generate Maps';
        elements.btnRunBenchmark.disabled = false;
    }
}

function update2DMapPreviews() {
    if (!state.currentMapData) return;

    const maps = state.currentMapData.maps;
    const hasGT = !!maps.gt_normal;

    if (hasGT) {
        elements.toggleGtBtn.classList.remove('hidden');
        elements.gtToggleLabel.textContent = state.showingGT ? 'Show Predicted Maps' : 'Show GT Maps';
    } else {
        elements.toggleGtBtn.classList.add('hidden');
    }

    elements.imgInputRgb.src = maps.input_rgb;

    if (state.showingGT && hasGT) {
        elements.imgNormalMap.src = maps.gt_normal;
        elements.imgRoughnessMap.src = maps.gt_roughness || maps.roughness;
        elements.imgHeightMap.src = maps.gt_height || maps.height;

        elements.titleNormal.textContent = "Normal Map (Ground Truth)";
        elements.titleRoughness.textContent = "Roughness Map (Ground Truth)";
        elements.titleHeight.textContent = "Height Map (Ground Truth)";
        elements.badgeNormalTech.textContent = "Ground Truth";
    } else {
        const showUNet = state.activeNormalView === 'unet' && maps.normal_unet;
        elements.imgNormalMap.src = showUNet ? maps.normal_unet : (maps.normal_dip || maps.normal);
        elements.imgRoughnessMap.src = maps.roughness;
        elements.imgHeightMap.src = maps.height;

        elements.titleNormal.textContent = showUNet ? "Normal Map (PyTorch U-Net)" : "Normal Map (Classical DIP)";
        elements.badgeNormalTech.textContent = showUNet ? "PyTorch U-Net" : "Sobel Gradients";
        elements.titleRoughness.textContent = "Roughness Map (2D DFT)";
        elements.titleHeight.textContent = "Height Map (Poisson Integration)";
    }
}

function updateMetricsScorecard(metrics) {
    if (!metrics) return;

    const unet = metrics.normal_unet || {};
    const dip = metrics.normal_dip || {};
    const rough = metrics.roughness || {};
    const height = metrics.height || {};

    const hasGT = unet.mae !== null && unet.mae !== undefined;

    if (hasGT) {
        elements.gtStatusBadge.textContent = "GT Scorecard Active";
        elements.gtStatusBadge.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-700/50 text-emerald-400 font-mono";

        if (state.method === 'both') {
            elements.metricMae.textContent = `U-Net ${unet.mae}° | DIP ${dip.mae}°`;
            elements.metricMaeSub.textContent = `U-Net vs DIP MAE`;

            elements.metricPct1125.textContent = `${unet.pct_11_25}% | ${dip.pct_11_25}%`;
            elements.metricPctSub.textContent = `U-Net vs DIP %<11.25°`;

            elements.metricPct225.textContent = `${unet.pct_22_5}% | ${dip.pct_22_5}%`;
            elements.metricNormalPsnr.textContent = `${unet.psnr} | ${dip.psnr} dB`;
        } else if (state.method === 'unet') {
            elements.metricMae.textContent = `${unet.mae}°`;
            elements.metricMaeSub.textContent = `PyTorch U-Net MAE`;
            elements.metricPct1125.textContent = `${unet.pct_11_25}%`;
            elements.metricPctSub.textContent = `Higher is better`;
            elements.metricPct225.textContent = `${unet.pct_22_5}%`;
            elements.metricNormalPsnr.textContent = `${unet.psnr} dB`;
        } else {
            elements.metricMae.textContent = `${dip.mae}°`;
            elements.metricMaeSub.textContent = `Classical DIP MAE`;
            elements.metricPct1125.textContent = `${dip.pct_11_25}%`;
            elements.metricPctSub.textContent = `Higher is better`;
            elements.metricPct225.textContent = `${dip.pct_22_5}%`;
            elements.metricNormalPsnr.textContent = `${dip.psnr} dB`;
        }

        elements.metricRoughSsim.textContent = `${rough.ssim || '--'}`;
        elements.metricHeightSsim.textContent = `${height.ssim || '--'}`;
    } else {
        elements.gtStatusBadge.textContent = "Live Mode (No GT)";
        elements.gtStatusBadge.className = "text-xs px-2.5 py-0.5 rounded-full bg-gray-800 border border-gray-700 text-gray-400 font-mono";

        elements.metricMae.textContent = '--°';
        elements.metricMaeSub.textContent = 'Lower is better';
        elements.metricPct1125.textContent = '--%';
        elements.metricPctSub.textContent = 'Higher is better';
        elements.metricPct225.textContent = '--%';
        elements.metricNormalPsnr.textContent = '-- dB';
        elements.metricRoughSsim.textContent = '--';
        elements.metricHeightSsim.textContent = '--';
    }
}

async function initThreeJS() {
    const container = elements.canvasContainer;
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 400;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070c);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.2);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const [vertSrc, fragSrc] = await Promise.all([
        fetch('/static/shaders/relight.vert').then(res => res.text()),
        fetch('/static/shaders/relight.frag').then(res => res.text())
    ]);

    const textureLoader = new THREE.TextureLoader();
    const fallbackCanvas = document.createElement('canvas');
    fallbackCanvas.width = fallbackCanvas.height = 128;
    const ctx = fallbackCanvas.getContext('2d');
    ctx.fillStyle = '#111726';
    ctx.fillRect(0, 0, 128, 128);
    const fallbackDataUrl = fallbackCanvas.toDataURL();

    defaultTextures.diffuse = textureLoader.load(fallbackDataUrl);
    defaultTextures.normal = textureLoader.load(fallbackDataUrl);
    defaultTextures.roughness = textureLoader.load(fallbackDataUrl);
    defaultTextures.height = textureLoader.load(fallbackDataUrl);

    shaderMaterial = new THREE.ShaderMaterial({
        vertexShader: vertSrc,
        fragmentShader: fragSrc,
        uniforms: {
            uDiffuseMap: { value: defaultTextures.diffuse },
            uNormalMap: { value: defaultTextures.normal },
            uRoughnessMap: { value: defaultTextures.roughness },
            uHeightMap: { value: defaultTextures.height },
            uLightPosition: { value: new THREE.Vector3(state.lightPos.x, state.lightPos.y, state.lightPos.z) },
            uLightColor: { value: new THREE.Vector3(1.0, 1.0, 1.0) },
            uAmbientIntensity: { value: 0.25 },
            uLightIntensity: { value: 1.2 },
            uSpecularStrength: { value: 1.0 },
            uRoughnessScale: { value: 1.0 },
            uDisplacementScale: { value: 0.15 }
        },
        side: THREE.DoubleSide
    });

    const geometry = new THREE.PlaneGeometry(2, 2, 128, 128);
    planeMesh = new THREE.Mesh(geometry, shaderMaterial);
    scene.add(planeMesh);

    initMouseLightControls(container);

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

function initMouseLightControls(container) {
    const onMove = (clientX, clientY) => {
        const rect = container.getBoundingClientRect();
        const normX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const normY = -(((clientY - rect.top) / rect.height) * 2 - 1);

        state.lightPos.x = normX * 2.0;
        state.lightPos.y = normY * 2.0;

        shaderMaterial.uniforms.uLightPosition.value.set(
            state.lightPos.x,
            state.lightPos.y,
            state.lightPos.z
        );

        elements.lightPosReadout.textContent = `L: (${state.lightPos.x.toFixed(1)}, ${state.lightPos.y.toFixed(1)}, ${state.lightPos.z.toFixed(1)})`;
    };

    container.addEventListener('mousedown', (e) => {
        state.isDraggingLight = true;
        onMove(e.clientX, e.clientY);
    });

    window.addEventListener('mousemove', (e) => {
        if (state.isDraggingLight) {
            onMove(e.clientX, e.clientY);
        }
    });

    window.addEventListener('mouseup', () => {
        state.isDraggingLight = false;
    });

    elements.btnResetLight.addEventListener('click', () => {
        state.lightPos = { x: 0.0, y: 0.0, z: 1.5 };
        elements.sliderLightZ.value = 1.5;
        elements.valLightZ.textContent = '1.5';

        shaderMaterial.uniforms.uLightPosition.value.set(0.0, 0.0, 1.5);
        elements.lightPosReadout.textContent = `L: (0.0, 0.0, 1.5)`;
    });
}

function initSliderControls() {
    elements.sliderLightZ.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state.lightPos.z = val;
        elements.valLightZ.textContent = val.toFixed(1);
        shaderMaterial.uniforms.uLightPosition.value.z = val;
        elements.lightPosReadout.textContent = `L: (${state.lightPos.x.toFixed(1)}, ${state.lightPos.y.toFixed(1)}, ${state.lightPos.z.toFixed(1)})`;
    });

    elements.sliderLightIntensity.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        elements.valLightIntensity.textContent = val.toFixed(1);
        shaderMaterial.uniforms.uLightIntensity.value = val;
    });

    elements.sliderSpecular.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        elements.valSpecular.textContent = val.toFixed(1);
        shaderMaterial.uniforms.uSpecularStrength.value = val;
    });

    elements.sliderDisplacement.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        elements.valDisplacement.textContent = val.toFixed(2);
        shaderMaterial.uniforms.uDisplacementScale.value = val;
    });

    elements.colorLight.addEventListener('input', (e) => {
        const hex = e.target.value;
        const color = new THREE.Color(hex);
        shaderMaterial.uniforms.uLightColor.value.set(color.r, color.g, color.b);
    });
}

function update3DTextures(maps) {
    if (!shaderMaterial || !maps) return;

    const loader = new THREE.TextureLoader();

    loader.load(maps.input_rgb, (tex) => {
        shaderMaterial.uniforms.uDiffuseMap.value = tex;
    });

    const activeNormal = (state.activeNormalView === 'unet' && maps.normal_unet) ? maps.normal_unet : (maps.normal_dip || maps.normal);
    loader.load(activeNormal, (tex) => {
        shaderMaterial.uniforms.uNormalMap.value = tex;
    });

    loader.load(maps.roughness, (tex) => {
        shaderMaterial.uniforms.uRoughnessMap.value = tex;
    });

    loader.load(maps.height, (tex) => {
        shaderMaterial.uniforms.uHeightMap.value = tex;
    });
}

async function captureWebcamPhoto() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        await video.play();

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        stream.getTracks().forEach(track => track.stop());

        canvas.toBlob((blob) => {
            const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
            runMapGeneration({ file });
        }, 'image/jpeg');

    } catch (err) {
        alert('Could not access camera: ' + err.message);
    }
}
