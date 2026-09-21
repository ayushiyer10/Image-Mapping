/**
 * Single-Image Surface Map Estimation & Real-Time Physical Relighting
 * Frontend Application JavaScript
 * Traditional DIP (Sobel + 2D DFT) vs Deep Learning (MiDaS_small)
 */

const state = {
    activeMethodView: 'both', // 'both', 'dip', or 'midas'
    currentMapData: null,
    benchmarkData: null,
    lightPos: { x: 0.0, y: 0.0, z: 1.5 },
    isDraggingLight: false
};

let scene, camera, renderer, planeMesh, shaderMaterial;
let defaultTextures = {};

const elements = {
    fileInput: document.getElementById('file-input'),
    fileNameLabel: document.getElementById('file-name-label'),
    btnWebcam: document.getElementById('btn-webcam'),
    btnGenerate: document.getElementById('btn-generate'),
    btnGenerateText: document.getElementById('btn-generate-text'),

    // Zip and Map Download Buttons
    btnDownloadZip: document.getElementById('btn-download-zip'),
    btnDownloadZipText: document.getElementById('btn-download-zip-text'),
    btnDownloadInputRgb: document.getElementById('btn-download-input-rgb'),
    btnDownloadNormalDip: document.getElementById('btn-download-normal-dip'),
    btnDownloadNormalDl: document.getElementById('btn-download-normal-dl'),
    btnDownloadRoughness: document.getElementById('btn-download-roughness'),
    btnDownloadHeight: document.getElementById('btn-download-height'),

    // Toggle Buttons
    toggleBoth: document.getElementById('toggle-both'),
    toggleDip: document.getElementById('toggle-dip'),
    toggleMidas: document.getElementById('toggle-midas'),

    // Error Notification
    errorBanner: document.getElementById('error-banner'),
    errorMessage: document.getElementById('error-message'),
    btnDismissError: document.getElementById('btn-dismiss-error'),

    // 2D Map Display
    imgInputRgb: document.getElementById('img-input-rgb'),
    normalSectionHeading: document.getElementById('normal-section-heading'),
    normalMapsContainer: document.getElementById('normal-maps-container'),
    cardNormalDip: document.getElementById('card-normal-dip'),
    cardNormalDl: document.getElementById('card-normal-dl'),
    imgNormalDip: document.getElementById('img-normal-dip'),
    imgNormalDl: document.getElementById('img-normal-dl'),
    imgRoughnessMap: document.getElementById('img-roughness-map'),
    imgHeightMap: document.getElementById('img-height-map'),
    badgeSourceType: document.getElementById('badge-source-type'),

    // 3D Canvas & Relighting
    canvasContainer: document.getElementById('canvas-container'),
    btnResetLight: document.getElementById('btn-reset-light'),
    lightPosReadout: document.getElementById('light-pos-readout'),

    sliderLightZ: document.getElementById('slider-light-z'),
    sliderLightIntensity: document.getElementById('slider-light-intensity'),
    sliderSpecular: document.getElementById('slider-specular'),
    sliderDisplacement: document.getElementById('slider-displacement'),

    valLightZ: document.getElementById('val-light-z'),
    valLightIntensity: document.getElementById('val-light-intensity'),
    valSpecular: document.getElementById('val-specular'),
    valDisplacement: document.getElementById('val-displacement'),

    // Scorecard Sections
    scorecardConfidenceWrapper: document.getElementById('scorecard-confidence-wrapper'),
    scorecardGtWrapper: document.getElementById('scorecard-gt-wrapper'),

    // Confidence Scorecard Elements
    confNormalVal: document.getElementById('conf-normal-val'),
    confNormalBand: document.getElementById('conf-normal-band'),
    confNormalDesc: document.getElementById('conf-normal-desc'),

    confHeightVal: document.getElementById('conf-height-val'),
    confHeightBand: document.getElementById('conf-height-band'),
    confHeightDesc: document.getElementById('conf-height-desc'),

    confRoughVal: document.getElementById('conf-rough-val'),
    confRoughBand: document.getElementById('conf-rough-band'),
    confRoughDesc: document.getElementById('conf-rough-desc'),
    confDisclaimerText: document.getElementById('conf-disclaimer-text'),

    // GT Scorecard Elements
    gtMetricMae: document.getElementById('gt-metric-mae'),
    gtMetricMaeSub: document.getElementById('gt-metric-mae-sub'),
    gtMetricPct11: document.getElementById('gt-metric-pct11'),
    gtMetricPsnr: document.getElementById('gt-metric-psnr'),
    gtMetricSsim: document.getElementById('gt-metric-ssim'),

    // Section 5: Benchmark Table
    benchmarkTableBody: document.getElementById('benchmark-table-body')
};

document.addEventListener('DOMContentLoaded', async () => {
    initEventListeners();
    initMethodToggle();
    initSliderControls();
    await initThreeJS();
    
    // Fetch benchmark results in background
    fetchBenchmark();
});

function showError(message) {
    if (elements.errorBanner && elements.errorMessage) {
        elements.errorMessage.textContent = message || "An unexpected error occurred.";
        elements.errorBanner.classList.remove('hidden');
    }
}

function hideError() {
    if (elements.errorBanner) {
        elements.errorBanner.classList.add('hidden');
    }
}

function initEventListeners() {
    if (elements.btnDismissError) {
        elements.btnDismissError.addEventListener('click', hideError);
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const selectedFile = e.target.files[0];
                if (elements.fileNameLabel) {
                    elements.fileNameLabel.textContent = selectedFile.name;
                }
                runMapGeneration({ file: selectedFile });
            }
        });
    }

    if (elements.btnWebcam) {
        elements.btnWebcam.addEventListener('click', () => {
            captureWebcamPhoto();
        });
    }

    if (elements.btnGenerate) {
        elements.btnGenerate.addEventListener('click', () => {
            if (elements.fileInput && elements.fileInput.files && elements.fileInput.files[0]) {
                runMapGeneration({ file: elements.fileInput.files[0] });
            } else {
                showError("Please select an image file first.");
            }
        });
    }

    if (elements.btnDownloadZip) {
        elements.btnDownloadZip.addEventListener('click', downloadZipArchive);
    }

    if (elements.btnDownloadInputRgb) {
        elements.btnDownloadInputRgb.addEventListener('click', () => downloadSingleMap('input_rgb', '01_input_rgb.png'));
    }
    if (elements.btnDownloadNormalDip) {
        elements.btnDownloadNormalDip.addEventListener('click', () => downloadSingleMap('normal_dip', '02_normal_dip_classical.png'));
    }
    if (elements.btnDownloadNormalDl) {
        elements.btnDownloadNormalDl.addEventListener('click', () => downloadSingleMap('normal_dl', '03_normal_midas_dl.png'));
    }
    if (elements.btnDownloadRoughness) {
        elements.btnDownloadRoughness.addEventListener('click', () => downloadSingleMap('roughness', '04_roughness_map.png'));
    }
    if (elements.btnDownloadHeight) {
        elements.btnDownloadHeight.addEventListener('click', () => downloadSingleMap('height', '05_height_poisson_map.png'));
    }
}

function downloadSingleMap(mapKey, defaultFilename) {
    if (!state.currentMapData || !state.currentMapData.maps) {
        showError("No generated maps available for download. Please process an image first.");
        return;
    }
    const maps = state.currentMapData.maps;
    const mapUrl = maps[mapKey] || (mapKey === 'normal_dl' ? maps.normal : null);
    if (!mapUrl) {
        showError(`Map '${mapKey}' is not available.`);
        return;
    }
    const a = document.createElement('a');
    a.href = mapUrl;
    a.download = defaultFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

async function downloadZipArchive() {
    if (!state.currentMapData || !state.currentMapData.maps) {
        showError("No generated maps available to package into ZIP. Please process an image first.");
        return;
    }

    if (elements.btnDownloadZip) {
        elements.btnDownloadZip.disabled = true;
        if (elements.btnDownloadZipText) elements.btnDownloadZipText.textContent = "Packing ZIP...";
    }

    try {
        const resp = await fetch('/api/download_zip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ maps: state.currentMapData.maps })
        });

        if (!resp.ok) {
            let detail = 'Failed to generate ZIP file.';
            try {
                const errJson = await resp.json();
                if (errJson.detail) detail = errJson.detail;
            } catch (e) {}
            showError(detail);
            return;
        }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'surface_maps.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (err) {
        console.error('ZIP Download failed:', err);
        showError(`ZIP Download failed: ${err.message || err}`);
    } finally {
        if (elements.btnDownloadZip) {
            elements.btnDownloadZip.disabled = false;
            if (elements.btnDownloadZipText) elements.btnDownloadZipText.textContent = "Download All Maps (.zip)";
        }
    }
}

function initMethodToggle() {
    const setMethod = (method) => {
        state.activeMethodView = method;

        // Button styles
        const activeClass = "flex-1 py-1.5 rounded-lg font-semibold transition text-white bg-indigo-600 shadow";
        const inactiveClass = "flex-1 py-1.5 rounded-lg font-medium transition text-gray-400 hover:text-white";

        if (elements.toggleBoth) elements.toggleBoth.className = method === 'both' ? activeClass : inactiveClass;
        if (elements.toggleDip) elements.toggleDip.className = method === 'dip' ? activeClass : inactiveClass;
        if (elements.toggleMidas) elements.toggleMidas.className = method === 'midas' ? activeClass : inactiveClass;

        // Card visibility & layout
        if (method === 'both') {
            if (elements.cardNormalDip) elements.cardNormalDip.classList.remove('hidden');
            if (elements.cardNormalDl) elements.cardNormalDl.classList.remove('hidden');
            if (elements.normalMapsContainer) elements.normalMapsContainer.className = "grid grid-cols-1 sm:grid-cols-2 gap-3";
            if (elements.normalSectionHeading) elements.normalSectionHeading.textContent = "Normal Map: Traditional DIP vs Deep Learning (MiDaS)";
        } else if (method === 'dip') {
            if (elements.cardNormalDip) elements.cardNormalDip.classList.remove('hidden');
            if (elements.cardNormalDl) elements.cardNormalDl.classList.add('hidden');
            if (elements.normalMapsContainer) elements.normalMapsContainer.className = "grid grid-cols-1 gap-3";
            if (elements.normalSectionHeading) elements.normalSectionHeading.textContent = "Normal Map: Traditional DIP (Sobel)";
        } else {
            if (elements.cardNormalDip) elements.cardNormalDip.classList.add('hidden');
            if (elements.cardNormalDl) elements.cardNormalDl.classList.remove('hidden');
            if (elements.normalMapsContainer) elements.normalMapsContainer.className = "grid grid-cols-1 gap-3";
            if (elements.normalSectionHeading) elements.normalSectionHeading.textContent = "Normal Map: Deep Learning (MiDaS)";
        }

        if (state.currentMapData) {
            update3DTextures(state.currentMapData.maps);
        }
    };

    if (elements.toggleBoth) elements.toggleBoth.addEventListener('click', () => setMethod('both'));
    if (elements.toggleDip) elements.toggleDip.addEventListener('click', () => setMethod('dip'));
    if (elements.toggleMidas) elements.toggleMidas.addEventListener('click', () => setMethod('midas'));
}

async function runMapGeneration({ file = null, assetId = null }) {
    hideError();

    if (elements.btnGenerate) {
        elements.btnGenerate.disabled = true;
        if (elements.btnGenerateText) elements.btnGenerateText.textContent = "Estimating...";
    }

    try {
        const formData = new FormData();
        formData.append('method', 'both');

        if (file) {
            formData.append('file', file);
        } else if (assetId) {
            formData.append('asset_id', assetId);
        } else {
            showError("Please choose an image file.");
            return;
        }

        const resp = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            let detail = 'Failed to process image.';
            try {
                const errorJson = await resp.json();
                if (errorJson.detail) detail = errorJson.detail;
            } catch (e) {}
            showError(`Error: ${detail}`);
            return;
        }

        const data = await resp.json();
        state.currentMapData = data;

        update2DMapPreviews(data);
        updateScorecard(data);
        update3DTextures(data.maps);

    } catch (err) {
        console.error('Surface Map estimation failed:', err);
        showError(`Surface Map estimation failed: ${err.message || err}`);
    } finally {
        if (elements.btnGenerate) {
            elements.btnGenerate.disabled = false;
            if (elements.btnGenerateText) elements.btnGenerateText.textContent = "Generate Maps";
        }
    }
}

function update2DMapPreviews(data) {
    if (!data || !data.maps) return;

    const maps = data.maps;
    if (elements.imgInputRgb && maps.input_rgb) elements.imgInputRgb.src = maps.input_rgb;
    if (elements.imgNormalDip && (maps.normal_dip || maps.normal)) elements.imgNormalDip.src = maps.normal_dip || maps.normal;
    if (elements.imgNormalDl && (maps.normal_dl || maps.normal)) elements.imgNormalDl.src = maps.normal_dl || maps.normal;
    if (elements.imgRoughnessMap && maps.roughness) elements.imgRoughnessMap.src = maps.roughness;
    if (elements.imgHeightMap && maps.height) elements.imgHeightMap.src = maps.height;

    if (elements.badgeSourceType) {
        elements.badgeSourceType.textContent = data.has_ground_truth ? "Dataset Reference (GT ✓)" : "Custom Photo Upload";
        elements.badgeSourceType.className = data.has_ground_truth ?
            "text-xs px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 font-mono border border-emerald-800/40" :
            "text-xs px-2.5 py-0.5 rounded-full bg-indigo-950 text-indigo-400 font-mono border border-indigo-800/40";
    }
}

function applyBandBadgeStyle(element, band) {
    if (!element) return;
    const b = (band || 'medium').toLowerCase();
    element.textContent = b.toUpperCase();

    if (b === 'high') {
        element.className = "text-[10px] px-2 py-0.5 rounded-full font-mono uppercase font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-700/50";
    } else if (b === 'low') {
        element.className = "text-[10px] px-2 py-0.5 rounded-full font-mono uppercase font-semibold bg-rose-950/80 text-rose-400 border border-rose-700/50";
    } else {
        element.className = "text-[10px] px-2 py-0.5 rounded-full font-mono uppercase font-semibold bg-amber-950/80 text-amber-400 border border-amber-700/50";
    }
}

function updateScorecard(data) {
    if (!data) return;

    const hasGT = !!data.has_ground_truth;

    if (hasGT) {
        // Show GT Scorecard, hide Confidence Scorecard
        if (elements.scorecardGtWrapper) elements.scorecardGtWrapper.classList.remove('hidden');
        if (elements.scorecardConfidenceWrapper) elements.scorecardConfidenceWrapper.classList.add('hidden');

        const metrics = data.metrics || {};
        const dl = metrics.normal_dl || metrics.normal || {};
        const dip = metrics.normal_dip || {};
        const rough = metrics.roughness || {};
        const height = metrics.height || {};

        if (elements.gtMetricMae) {
            elements.gtMetricMae.textContent = (dl.mae !== null && dl.mae !== undefined) ? `${dl.mae}°` : (dip.mae !== null ? `${dip.mae}°` : 'N/A');
        }
        if (elements.gtMetricMaeSub) {
            elements.gtMetricMaeSub.textContent = `MiDaS DL MAE (vs DIP: ${dip.mae ?? 'N/A'}°)`;
        }
        if (elements.gtMetricPct11) {
            elements.gtMetricPct11.textContent = (dl.pct_11_25 !== null && dl.pct_11_25 !== undefined) ? `${dl.pct_11_25}%` : 'N/A';
        }
        if (elements.gtMetricPsnr) {
            elements.gtMetricPsnr.textContent = (dl.psnr !== null && dl.psnr !== undefined) ? `${dl.psnr} dB` : 'N/A';
        }
        if (elements.gtMetricSsim) {
            const r_ssim = (rough.ssim !== null && rough.ssim !== undefined) ? rough.ssim : 'N/A';
            const h_ssim = (height.ssim !== null && height.ssim !== undefined) ? height.ssim : 'N/A';
            elements.gtMetricSsim.textContent = `R: ${r_ssim} | H: ${h_ssim}`;
        }
    } else {
        // Live Upload (No GT): Show Confidence Scorecard, hide GT Scorecard
        if (elements.scorecardConfidenceWrapper) elements.scorecardConfidenceWrapper.classList.remove('hidden');
        if (elements.scorecardGtWrapper) elements.scorecardGtWrapper.classList.add('hidden');

        const conf = data.confidence || {};

        if (elements.confNormalVal) {
            elements.confNormalVal.textContent = (conf.normal && conf.normal.confidence_pct !== undefined) ? `${conf.normal.confidence_pct}%` : 'N/A';
        }
        applyBandBadgeStyle(elements.confNormalBand, conf.normal?.confidence_band);
        if (elements.confNormalDesc) {
            elements.confNormalDesc.textContent = conf.normal?.description || 'Angular vector similarity between Classical DIP and MiDaS Deep Learning normal estimates.';
        }

        if (elements.confHeightVal) {
            elements.confHeightVal.textContent = (conf.height && conf.height.confidence_pct !== undefined) ? `${conf.height.confidence_pct}%` : 'N/A';
        }
        applyBandBadgeStyle(elements.confHeightBand, conf.height?.confidence_band);
        if (elements.confHeightDesc) {
            elements.confHeightDesc.textContent = conf.height?.description || 'Closed-loop Poisson height integration consistency check.';
        }

        if (elements.confRoughVal) {
            elements.confRoughVal.textContent = (conf.roughness && conf.roughness.confidence_pct !== undefined) ? `${conf.roughness.confidence_pct}%` : 'N/A';
        }
        applyBandBadgeStyle(elements.confRoughBand, conf.roughness?.confidence_band);
        if (elements.confRoughDesc) {
            elements.confRoughDesc.textContent = conf.roughness?.description || 'SSIM structural stability across 2D DFT frequency cutoff radii.';
        }

        if (elements.confDisclaimerText) {
            elements.confDisclaimerText.textContent = conf.disclaimer || 'Ground-truth unavailable for custom uploaded images. Scores reflect algorithm consistency and structural stability.';
        }
    }
}

async function fetchBenchmark() {
    try {
        const resp = await fetch('/api/benchmark');
        if (!resp.ok) return;
        const data = await resp.json();
        state.benchmarkData = data;
        renderBenchmarkTable(data);
    } catch (err) {
        console.error('Failed to fetch benchmark results:', err);
        if (elements.benchmarkTableBody) {
            elements.benchmarkTableBody.innerHTML = `<tr><td colspan="8" class="p-4 text-center text-rose-400">Failed to load benchmark data. Run benchmark.py to generate data.</td></tr>`;
        }
    }
}

function renderBenchmarkTable(data) {
    if (!elements.benchmarkTableBody || !data || !data.summary) return;

    let html = '';
    const summary = data.summary;

    Object.keys(summary).forEach(sourceKey => {
        const item = summary[sourceKey];
        if (!item.normal_dip || !item.normal_dl) return;

        const sourceLabel = sourceKey.toUpperCase();
        const count = item.count || 0;

        const dip = item.normal_dip;
        const dl = item.normal_dl;

        const fmt = (val, suffix = '') => (val !== null && val !== undefined) ? `${val}${suffix}` : 'N/A';

        html += `
            <tr class="border-b border-cardborder/40 hover:bg-white/[0.02] transition">
                <td class="p-3 font-semibold text-gray-300" rowspan="2">
                    <div class="font-bold text-gray-200">${sourceLabel}</div>
                    <div class="text-[10px] text-gray-500 font-normal">N=${count} samples</div>
                </td>
                <td class="p-3 text-cyan-400 font-semibold flex items-center space-x-1.5">
                    <i class="fa-solid fa-wave-square text-xs"></i>
                    <span>Traditional DIP (Sobel)</span>
                </td>
                <td class="p-3 font-mono text-gray-300">${fmt(dip.mean_mae, '°')}</td>
                <td class="p-3 font-mono text-gray-400">${fmt(dip.median_mae, '°')}</td>
                <td class="p-3 font-mono text-gray-400">${fmt(dip.pct_11_25, '%')}</td>
                <td class="p-3 font-mono text-gray-400">${fmt(dip.pct_22_5, '%')}</td>
                <td class="p-3 font-mono text-gray-400">${fmt(dip.mean_psnr, ' dB')}</td>
                <td class="p-3 font-mono text-gray-400">${fmt(dip.mean_ssim)}</td>
            </tr>
            <tr class="border-b border-cardborder/80 bg-purple-950/20 hover:bg-purple-950/30 transition">
                <td class="p-3 text-purple-400 font-bold flex items-center space-x-1.5">
                    <i class="fa-solid fa-brain text-xs"></i>
                    <span>Deep Learning (MiDaS)</span>
                </td>
                <td class="p-3 font-mono text-emerald-400 font-bold">${fmt(dl.mean_mae, '°')}</td>
                <td class="p-3 font-mono text-emerald-400 font-semibold">${fmt(dl.median_mae, '°')}</td>
                <td class="p-3 font-mono text-emerald-400 font-semibold">${fmt(dl.pct_11_25, '%')}</td>
                <td class="p-3 font-mono text-emerald-400 font-semibold">${fmt(dl.pct_22_5, '%')}</td>
                <td class="p-3 font-mono text-emerald-400 font-semibold">${fmt(dl.mean_psnr, ' dB')}</td>
                <td class="p-3 font-mono text-emerald-400 font-semibold">${fmt(dl.mean_ssim)}</td>
            </tr>
        `;
    });

    elements.benchmarkTableBody.innerHTML = html || `<tr><td colspan="8" class="p-4 text-center text-gray-500">No benchmark summary data found.</td></tr>`;
}

async function initThreeJS() {
    const container = elements.canvasContainer;
    if (!container) return;

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

        if (elements.lightPosReadout) {
            elements.lightPosReadout.textContent = `L: (${state.lightPos.x.toFixed(1)}, ${state.lightPos.y.toFixed(1)}, ${state.lightPos.z.toFixed(1)})`;
        }
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

    if (elements.btnResetLight) {
        elements.btnResetLight.addEventListener('click', () => {
            state.lightPos = { x: 0.0, y: 0.0, z: 1.5 };
            if (elements.sliderLightZ) elements.sliderLightZ.value = 1.5;
            if (elements.valLightZ) elements.valLightZ.textContent = '1.5';

            shaderMaterial.uniforms.uLightPosition.value.set(0.0, 0.0, 1.5);
            if (elements.lightPosReadout) elements.lightPosReadout.textContent = `L: (0.0, 0.0, 1.5)`;
        });
    }
}

function initSliderControls() {
    if (elements.sliderLightZ) {
        elements.sliderLightZ.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            state.lightPos.z = val;
            if (elements.valLightZ) elements.valLightZ.textContent = val.toFixed(1);
            shaderMaterial.uniforms.uLightPosition.value.z = val;
            if (elements.lightPosReadout) {
                elements.lightPosReadout.textContent = `L: (${state.lightPos.x.toFixed(1)}, ${state.lightPos.y.toFixed(1)}, ${state.lightPos.z.toFixed(1)})`;
            }
        });
    }

    if (elements.sliderLightIntensity) {
        elements.sliderLightIntensity.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (elements.valLightIntensity) elements.valLightIntensity.textContent = val.toFixed(1);
            shaderMaterial.uniforms.uLightIntensity.value = val;
        });
    }

    if (elements.sliderSpecular) {
        elements.sliderSpecular.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (elements.valSpecular) elements.valSpecular.textContent = val.toFixed(1);
            shaderMaterial.uniforms.uSpecularStrength.value = val;
        });
    }

    if (elements.sliderDisplacement) {
        elements.sliderDisplacement.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (elements.valDisplacement) elements.valDisplacement.textContent = val.toFixed(2);
            shaderMaterial.uniforms.uDisplacementScale.value = val;
        });
    }
}

function update3DTextures(maps) {
    if (!shaderMaterial || !maps) return;

    const loader = new THREE.TextureLoader();

    if (maps.input_rgb) {
        loader.load(maps.input_rgb, (tex) => {
            shaderMaterial.uniforms.uDiffuseMap.value = tex;
        });
    }

    const activeNormal = (state.activeMethodView === 'dip' && maps.normal_dip) ? maps.normal_dip : (maps.normal_dl || maps.normal);
    if (activeNormal) {
        loader.load(activeNormal, (tex) => {
            shaderMaterial.uniforms.uNormalMap.value = tex;
        });
    }

    if (maps.roughness) {
        loader.load(maps.roughness, (tex) => {
            shaderMaterial.uniforms.uRoughnessMap.value = tex;
        });
    }

    if (maps.height) {
        loader.load(maps.height, (tex) => {
            shaderMaterial.uniforms.uHeightMap.value = tex;
        });
    }
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
            if (elements.fileNameLabel) elements.fileNameLabel.textContent = "Webcam Capture.jpg";
            runMapGeneration({ file });
        }, 'image/jpeg');

    } catch (err) {
        showError('Could not access camera: ' + err.message);
    }
}
