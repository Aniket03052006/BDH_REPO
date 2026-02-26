import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ═══════════════════════════════════════════
// CONFIGURATION — set dynamically from server
// ═══════════════════════════════════════════
let NEURON_COUNT = 16384; // default, will be updated from server config
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
// MULTI-SERVER LOAD BALANCING (Free Tier Hack)
// List all your Render/Railway/HF URLs here
const BACKEND_HOSTS = [
    'bdh-repo-n9hl.onrender.com',
    'bdh-repo-9ein.onrender.com',
    'bdh-repo-yosc.onrender.com'
];

// Simple Round Robin: Pick a random one or stick to the first successful one
function getBestHost() {
    if (BACKEND_HOSTS.length === 0) return (window.location.port ? `${window.location.hostname}:${window.location.port}` : window.location.host);
    // For now, pick a random one to distribute load
    return BACKEND_HOSTS[Math.floor(Math.random() * BACKEND_HOSTS.length)];
}

const host = getBestHost();
const WS_URL = `${protocol}//${host}/ws`;

// Monosemantic color gradient: black → blue → cyan → yellow → white
const COLOR_STOPS = [
    { t: 0.00, color: new THREE.Color(0x0a0a12) },
    { t: 0.15, color: new THREE.Color(0x1a237e) },
    { t: 0.35, color: new THREE.Color(0x0288d1) },
    { t: 0.55, color: new THREE.Color(0x00bcd4) },
    { t: 0.75, color: new THREE.Color(0xfdd835) },
    { t: 1.00, color: new THREE.Color(0xffffff) }
];

function getColorForActivation(val) {
    if (val <= 0) return COLOR_STOPS[0].color.clone();
    const v = Math.min(1, val);
    for (let i = 1; i < COLOR_STOPS.length; i++) {
        if (v <= COLOR_STOPS[i].t) {
            const ratio = (v - COLOR_STOPS[i - 1].t) / (COLOR_STOPS[i].t - COLOR_STOPS[i - 1].t);
            return COLOR_STOPS[i - 1].color.clone().lerp(COLOR_STOPS[i].color, ratio);
        }
    }
    return COLOR_STOPS[COLOR_STOPS.length - 1].color.clone();
}

// ═══════════════════════════════════════════
// THREE.JS SETUP
// ═══════════════════════════════════════════
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const container = document.getElementById('canvas-container');
let width = container.clientWidth || window.innerWidth * 0.5;
let height = container.clientHeight || window.innerHeight;

const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
camera.position.z = 25;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;

const resizeObserver = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
});
resizeObserver.observe(container);

// ═══════════════════════════════════════════
// NEURON GLOBE — Fibonacci sphere
// ═══════════════════════════════════════════
let geometry, positions, colors, sizes, material, points;
let currentActivation, targetActivation;

// Synapse lines (geodesic arcs)
let synapseGroup = new THREE.Group();
scene.add(synapseGroup);
const MAX_SYNAPSE_ARCS = 30;
const ARC_SEGMENTS = 16; // sample points per arc
const SPHERE_RADIUS = 10; // must match globe radius

// SLERP: spherical linear interpolation for great-circle arcs
function slerp(v1, v2, t) {
    const dot = Math.max(-1, Math.min(1, v1.dot(v2)));
    const theta = Math.acos(dot);
    if (theta < 0.001) {
        // Vectors nearly identical, lerp instead
        return v1.clone().lerp(v2, t);
    }
    const sinTheta = Math.sin(theta);
    const a = Math.sin((1 - t) * theta) / sinTheta;
    const b = Math.sin(t * theta) / sinTheta;
    return v1.clone().multiplyScalar(a).add(v2.clone().multiplyScalar(b));
}

function createGeodesicArc(p1, p2, score, maxScore) {
    // Normalize to unit sphere, then SLERP, then scale back
    const n1 = p1.clone().normalize();
    const n2 = p2.clone().normalize();

    const arcPositions = [];
    const arcColors = [];
    for (let i = 0; i <= ARC_SEGMENTS; i++) {
        const t = i / ARC_SEGMENTS;
        const point = slerp(n1, n2, t).multiplyScalar(SPHERE_RADIUS * 1.005); // slightly above surface
        arcPositions.push(point.x, point.y, point.z);

        // Color by score: strong yellow to faint yellow gradient
        const intensity = maxScore > 0 ? score / maxScore : 0.5;
        // Strong yellow: #fdd835 (1.0, 0.85, 0.2)
        // Faint yellow: (0.15, 0.12, 0.02)
        const r = 0.15 + intensity * 0.85;
        const g = 0.12 + intensity * 0.73;
        const b2 = 0.02 + intensity * 0.18;
        arcColors.push(r, g, b2);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(arcPositions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(arcColors, 3));

    const intensity = maxScore > 0 ? score / maxScore : 0.5;
    const mat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.12 + intensity * 0.35, // weak=0.12, strong=0.47
        blending: THREE.AdditiveBlending,
        linewidth: 1
    });

    const line = new THREE.Line(geo, mat);
    line.userData.baseOpacity = mat.opacity;
    return line;
}

function initializeGlobe(neuronCount) {
    NEURON_COUNT = neuronCount;

    // Remove old globe if exists
    if (points) scene.remove(points);

    geometry = new THREE.BufferGeometry();
    positions = new Float32Array(NEURON_COUNT * 3);
    colors = new Float32Array(NEURON_COUNT * 3);
    sizes = new Float32Array(NEURON_COUNT);

    const phi = Math.PI * (3 - Math.sqrt(5));

    for (let i = 0; i < NEURON_COUNT; i++) {
        const y = 1 - (i / (NEURON_COUNT - 1)) * 2;
        const radiusAtY = Math.sqrt(1 - y * y);
        const theta = phi * i;

        positions[i * 3] = Math.cos(theta) * radiusAtY * 10;
        positions[i * 3 + 1] = y * 10;
        positions[i * 3 + 2] = Math.sin(theta) * radiusAtY * 10;

        colors[i * 3] = COLOR_STOPS[0].color.r;
        colors[i * 3 + 1] = COLOR_STOPS[0].color.g;
        colors[i * 3 + 2] = COLOR_STOPS[0].color.b;

        sizes[i] = 0.12;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    material = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending
    });

    points = new THREE.Points(geometry, material);
    scene.add(points);

    currentActivation = new Float32Array(NEURON_COUNT);
    targetActivation = new Float32Array(NEURON_COUNT);

    // Update label
    const label = document.getElementById('neuron-count-label');
    if (label) label.textContent = `${NEURON_COUNT.toLocaleString()} neurons`;

    // Remove loading
    setTimeout(() => {
        const loader = document.getElementById('loading-overlay');
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 500);
        }
    }, 300);
}

// Initialize with default count; will be re-initialized from server config
initializeGlobe(NEURON_COUNT);

// ═══════════════════════════════════════════
// CORE DATA STRUCTURES
// ═══════════════════════════════════════════
let xyTimeline = [];
let tokenChars = [];
let wordSpans = [];
let currentDisplayIndex = -1;
let playbackMode = 'live';
let selectedNeuronId = null;
let lastProfileTimeline = null; // cache for sparkline re-rendering on scrub
let isGenerating = false;

// ═══════════════════════════════════════════
// NEURON INTERACTION — Raycaster
// ═══════════════════════════════════════════
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.4;
const mouse = new THREE.Vector2();

const tooltip = document.getElementById('neuron-tooltip');
const tooltipId = document.getElementById('tooltip-id');
const tooltipValue = document.getElementById('tooltip-value');

function onMouseMove(event) {
    if (!points) return;
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);

    if (intersects.length > 0) {
        const idx = intersects[0].index;
        const act = currentActivation[idx];
        tooltipId.textContent = `Neuron #${idx}`;
        tooltipValue.textContent = act.toFixed(4);
        tooltip.style.left = `${event.clientX - rect.left + 15}px`;
        tooltip.style.top = `${event.clientY - rect.top - 10}px`;
        tooltip.classList.remove('hidden');
        container.style.cursor = 'pointer';
    } else {
        tooltip.classList.add('hidden');
        container.style.cursor = 'default';
    }
}

function onMouseClick(event) {
    if (!points) return;
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);

    if (intersects.length > 0) {
        const idx = intersects[0].index;

        // Shift+click = toggle ablation
        if (event.shiftKey) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: "toggle_ablation", index: idx }));
            }
            return;
        }

        // Normal click = select for profiling
        selectedNeuronId = idx;
        requestNeuronProfile(idx);
        updateLiveMonitor();
    }
}

container.addEventListener('mousemove', onMouseMove);
container.addEventListener('click', onMouseClick);

// ═══════════════════════════════════════════
// GLOBE COLOR UPDATE
// ═══════════════════════════════════════════
let ablatedSet = new Set();
let traceEnabled = true;
const DECAY_FACTOR = 0.96; // Hardcoded decay rate for Hebbian trace

function updateGlobeColors() {
    if (!geometry) return;
    const colorAttr = geometry.attributes.color;
    const sizeAttr = geometry.attributes.size;

    for (let i = 0; i < NEURON_COUNT; i++) {
        // If trace mode: blend toward target but also keep decaying glow
        if (traceEnabled) {
            // Approach target but never go below current decayed value
            const target = targetActivation[i];
            if (target > currentActivation[i]) {
                currentActivation[i] += (target - currentActivation[i]) * 0.3;
            } else {
                // Decay the old activation slowly (Hebbian trace glow)
                currentActivation[i] *= DECAY_FACTOR;
                // But still approach target if it's meaningful
                if (target > currentActivation[i]) {
                    currentActivation[i] = target;
                }
            }
        } else {
            // No trace: snap to target
            currentActivation[i] += (targetActivation[i] - currentActivation[i]) * 0.15;
        }

        if (ablatedSet.has(i)) {
            // Ablated neurons: red tint
            colorAttr.array[i * 3] = 0.5;
            colorAttr.array[i * 3 + 1] = 0.05;
            colorAttr.array[i * 3 + 2] = 0.05;
            sizeAttr.array[i] = 0.2;
        } else {
            const isTracing = traceEnabled && (currentActivation[i] > targetActivation[i] + 0.005);
            if (isTracing) {
                // Warm amber/orange trace for decaying neurons
                const v = Math.min(1, currentActivation[i]);
                colorAttr.array[i * 3] = 0.9 * v;       // Red channel: warm
                colorAttr.array[i * 3 + 1] = 0.4 * v;   // Green: amber tint
                colorAttr.array[i * 3 + 2] = 0.05 * v;  // Blue: very low
                sizeAttr.array[i] = 0.10 + Math.min(0.15, v * 0.15);
            } else {
                // Active neurons: normal blue/cyan/yellow gradient
                const color = getColorForActivation(currentActivation[i]);
                colorAttr.array[i * 3] = color.r;
                colorAttr.array[i * 3 + 1] = color.g;
                colorAttr.array[i * 3 + 2] = color.b;
                sizeAttr.array[i] = 0.10 + Math.min(0.25, currentActivation[i] * 0.3);
            }
        }
    }
    colorAttr.needsUpdate = true;
    sizeAttr.needsUpdate = true;
}

function updateTopActiveFromArray(xyArr) {
    // Build sorted top-10 from a raw activation array
    const candidates = [];
    for (let i = 0; i < xyArr.length; i++) {
        if (xyArr[i] > 0.01) candidates.push({ id: i, value: xyArr[i] });
    }
    candidates.sort((a, b) => b.value - a.value);
    const top10 = candidates.slice(0, 10);

    const topActiveEl = document.getElementById('np-top-active');
    if (!topActiveEl) return;
    if (top10.length === 0) {
        topActiveEl.innerHTML = '<span class="np-empty-hint">No active neurons at this token</span>';
        return;
    }
    topActiveEl.innerHTML = top10.map(n =>
        `<button class="np-top-active-chip${ablatedSet.has(n.id) ? ' ablated' : ''}" data-nid="${n.id}" title="Value: ${n.value.toFixed(4)}">#${n.id} <span class="np-top-active-val">${n.value.toFixed(2)}</span></button>`
    ).join('');
    topActiveEl.querySelectorAll('.np-top-active-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const nid = parseInt(chip.dataset.nid);
            selectedNeuronId = nid;
            requestNeuronProfile(nid);
        });
    });
}

function setGlobeFromTimeline(index) {
    if (index < 0 || index >= xyTimeline.length) return;
    const xy = xyTimeline[index];
    let maxVal = 0;
    for (let i = 0; i < NEURON_COUNT && i < xy.length; i++) {
        if (xy[i] > maxVal) maxVal = xy[i];
    }
    const scale = maxVal > 0 ? 1.0 / maxVal : 1.0;

    for (let i = 0; i < NEURON_COUNT; i++) {
        targetActivation[i] = i < xy.length ? xy[i] * scale : 0;
    }
    currentDisplayIndex = index;
    updatePlaybackLabel();
    updateLiveMonitor();
    updateTopActiveFromArray(xy);

    // Re-render sparkline with yellow peak at new position
    if (selectedNeuronId !== null && lastProfileTimeline) {
        renderSparkline(lastProfileTimeline, tokenChars);
    }
}

// Track current token index for clickable words
let generatedTokenStartIndex = 0;

// ═══════════════════════════════════════════
// CHAT UI
// ═══════════════════════════════════════════
const chatHistory = document.getElementById('chat-history');
let currentAiBubble = null;

function getTimeString() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function clearWelcome() {
    const welcome = chatHistory.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
}

function createUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row message-row--user';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar message-avatar--user';
    avatar.textContent = 'U';
    const content = document.createElement('div');
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-bubble--user';
    bubble.textContent = text;
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getTimeString();
    content.appendChild(bubble);
    content.appendChild(time);
    row.appendChild(avatar);
    row.appendChild(content);
    return row;
}

function createAiMessageRow() {
    const row = document.createElement('div');
    row.className = 'message-row message-row--ai';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar message-avatar--ai';
    avatar.innerHTML = `<img src="icon.png" alt="AI" style="width:100%;height:100%;object-fit:contain;border-radius:50%;">`;
    const content = document.createElement('div');
    content.className = 'message-content';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-bubble--ai';
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    bubble.appendChild(typing);
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getTimeString();
    content.appendChild(bubble);
    content.appendChild(time);
    row.appendChild(avatar);
    row.appendChild(content);
    return { row, bubble };
}

function appendToken(token) {
    clearWelcome();
    if (!currentAiBubble) {
        const { row, bubble } = createAiMessageRow();
        chatHistory.appendChild(row);
        currentAiBubble = bubble;
        generatedTokenStartIndex = xyTimeline.length - 1;
    }
    const typing = currentAiBubble.querySelector('.typing-indicator');
    if (typing) typing.remove();
    let textContainer = currentAiBubble.querySelector('.ai-text-content');
    if (!textContainer) {
        textContainer = document.createElement('span');
        textContainer.className = 'ai-text-content';
        currentAiBubble.appendChild(textContainer);
    }

    // Build clickable word spans instead of plain text
    const tokenIdx = xyTimeline.length - 1; // current token index
    if (token === ' ' || token === '\n' || token === '\t') {
        // Whitespace: just append as text node
        textContainer.appendChild(document.createTextNode(token === '\n' ? '\n' : token));
    } else {
        // Check if last child is a word span we can append to
        const lastChild = textContainer.lastChild;
        if (lastChild && lastChild.nodeType === 1 && lastChild.classList?.contains('chat-word')) {
            // Extend existing word span
            lastChild.textContent += token;
            lastChild.dataset.tokenEnd = tokenIdx;
        } else {
            // Create new word span
            const span = document.createElement('span');
            span.className = 'chat-word';
            span.textContent = token;
            span.dataset.tokenStart = tokenIdx;
            span.dataset.tokenEnd = tokenIdx;
            span.addEventListener('click', (e) => {
                const start = parseInt(e.target.dataset.tokenStart);
                const end = parseInt(e.target.dataset.tokenEnd);
                const mid = Math.floor((start + end) / 2);
                // Navigate globe + timeline to this word
                playbackMode = 'paused';
                const pauseBtn = document.getElementById('pause-btn');
                const playBtn = document.getElementById('play-btn');
                if (pauseBtn) pauseBtn.classList.add('active');
                if (playBtn) playBtn.classList.remove('active');
                setGlobeFromTimeline(mid);
                const slider = document.getElementById('timeline-slider');
                if (slider) slider.value = mid;
                // Highlight clicked word
                document.querySelectorAll('.chat-word.active-word').forEach(el => el.classList.remove('active-word'));
                e.target.classList.add('active-word');
            });
            textContainer.appendChild(span);
        }
    }

    // Only auto-scroll if user is near the bottom (not manually scrolled up)
    const isNearBottom = chatHistory.scrollHeight - chatHistory.scrollTop - chatHistory.clientHeight < 60;
    if (isNearBottom) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

function appendError(err) {
    const div = document.createElement('div');
    div.className = 'chat-error';
    div.textContent = err;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function resetChatUI() {
    chatHistory.innerHTML = '';
    const welcome = document.createElement('div');
    welcome.className = 'chat-welcome';
    welcome.innerHTML = `
        <div class="welcome-orb">
            <div class="orb-ring"></div>
            <img src="icon.png" class="orb-core" alt="Neural Core">
        </div>
        <h3 class="welcome-title">Neural Explorer</h3>
        <p class="welcome-desc">Enter a prompt to observe how semantic concept neurons fire in real-time. Click any neuron to understand what it represents.</p>
        <div class="welcome-chips">
            <button class="chip" data-prompt="The dog and a cat">The dog and a cat</button>
            <button class="chip" data-prompt="Once upon a time">Once upon a time</button>
            <button class="chip" data-prompt="The meaning of life">The meaning of life</button>
        </div>
    `;
    chatHistory.appendChild(welcome);
    bindChips();
    currentAiBubble = null;
}

// ═══════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════
let socket;
let reconnectInterval = 1000;

function connect() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        console.log('WS connected');
        reconnectInterval = 1000;
        const indicator = document.getElementById('fps-counter');
        if (indicator) indicator.innerText = 'Connected';
        const dot = document.querySelector('.status-dot');
        if (dot) {
            dot.style.background = '#14b8a6';
            dot.style.boxShadow = '0 0 6px rgba(20, 184, 166, 0.5)';
        }
    };

    socket.onmessage = (event) => {
        const data = jsonParseSafe(event.data);
        if (!data) return;

        if (data.error) {
            console.error('Server error:', data.error);
            appendError(data.error);
            isGenerating = false;
            return;
        }

        // Server config
        if (data.type === 'config') {
            const total = data.total_neurons;
            if (total && total !== NEURON_COUNT) {
                console.log(`Re-initializing globe for ${total} neurons`);
                initializeGlobe(total);
            }
            return;
        }

        if (data.status === 'reset') {
            xyTimeline = [];
            tokenChars = [];
            wordSpans = [];
            currentDisplayIndex = -1;
            targetActivation.fill(0);
            currentActivation.fill(0);
            isGenerating = false;
            selectedNeuronId = null;
            ablatedSet.clear();
            hideNeuronProfile();
            updatePlaybackLabel();
            return;
        }

        if (data.status === 'done') {
            currentAiBubble = null;
            isGenerating = false;
            return;
        }

        // Neuron profile response
        if (data.type === 'neuron_profile') {
            renderNeuronProfile(data);
            return;
        }

        // Ablation update
        if (data.type === 'ablation_update') {
            ablatedSet = new Set(data.ablation_list || []);
            updateAblationDisplay();
            return;
        }

        // Hebbian status
        if (data.type === 'hebb_status') {
            updateHebbStats(data.mem_max, data.mem_mean);
            return;
        }

        // Experiment B results
        if (data.type === 'experiment_result') {
            renderExperimentResult(data);
            return;
        }
        if (data.type === 'exp_phase' || data.type === 'exp_progress') {
            renderExperimentProgress(data);
            return;
        }

        // Token data packet
        if (data.type === 'token' && data.character !== undefined && data.xy_vis) {
            appendToken(data.character);

            const xyArr = new Float32Array(data.xy_vis);
            xyTimeline.push(xyArr);
            tokenChars.push(data.character);

            const slider = document.getElementById('timeline-slider');
            slider.max = xyTimeline.length - 1;

            if (playbackMode === 'live') {
                setGlobeFromTimeline(xyTimeline.length - 1);
                slider.value = xyTimeline.length - 1;
            }

            if (data.word_complete && data.word_string) {
                wordSpans.push({
                    word: data.word_string,
                    index: data.word_index
                });
            }

            // Update Hebb stats
            if (data.hebb_mem_max !== undefined) {
                updateHebbStats(data.hebb_mem_max, data.hebb_mem_mean);
            }
            // Update firing count
            if (data.firing_count !== undefined) {
                const fcLabel = document.getElementById('firing-count-label');
                if (fcLabel) fcLabel.textContent = `Firing: ${data.firing_count.toLocaleString()}`;
                const fsEl = document.getElementById('firing-stats');
                if (fsEl) fsEl.textContent = `Firing: ${data.firing_count.toLocaleString()} / ${NEURON_COUNT.toLocaleString()}`;
            }

            // Update top active neurons list
            if (data.top_neurons && data.top_neurons.length > 0) {
                const topActiveEl = document.getElementById('np-top-active');
                if (topActiveEl) {
                    topActiveEl.innerHTML = data.top_neurons.slice(0, 10).map(n =>
                        `<button class="np-top-active-chip${ablatedSet.has(n.id) ? ' ablated' : ''}" data-nid="${n.id}" title="Value: ${n.value}">#${n.id} <span class="np-top-active-val">${n.value.toFixed(2)}</span></button>`
                    ).join('');
                    topActiveEl.querySelectorAll('.np-top-active-chip').forEach(chip => {
                        chip.addEventListener('click', () => {
                            const nid = parseInt(chip.dataset.nid);
                            selectedNeuronId = nid;
                            requestNeuronProfile(nid);
                        });
                    });
                }
            }
        }
    };

    socket.onclose = () => {
        const indicator = document.getElementById('fps-counter');
        if (indicator) indicator.innerText = 'Reconnecting...';
        const dot = document.querySelector('.status-dot');
        if (dot) {
            dot.style.background = '#ef4444';
            dot.style.boxShadow = '0 0 6px rgba(239, 68, 68, 0.5)';
        }
        console.log(`WS closed, retrying in ${reconnectInterval}ms...`);
        setTimeout(connect, reconnectInterval);
        reconnectInterval = Math.min(30000, reconnectInterval * 2);
    };

    socket.onerror = (err) => {
        console.error('WS error:', err);
    };
}

function jsonParseSafe(str) {
    try { return JSON.parse(str); }
    catch (e) { return null; }
}

function requestNeuronProfile(neuronId) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "neuron_profile", neuron_id: neuronId }));
    }
}

// ═══════════════════════════════════════════
// NEURON PROFILE DASHBOARD
// ═══════════════════════════════════════════
function hideNeuronProfile() {
    document.getElementById('neuron-panel-empty').classList.remove('hidden');
    document.getElementById('neuron-profile').classList.add('hidden');
    clearSynapseLines();
}

function clearSynapseLines() {
    // Remove all arc objects from the group
    while (synapseGroup.children.length > 0) {
        const child = synapseGroup.children[0];
        child.geometry.dispose();
        child.material.dispose();
        synapseGroup.remove(child);
    }
}

function updateSynapseLines(sourceIdx, coFiringList) {
    if (!positions) return;
    clearSynapseLines();

    const sx = positions[sourceIdx * 3];
    const sy = positions[sourceIdx * 3 + 1];
    const sz = positions[sourceIdx * 3 + 2];
    const sourceVec = new THREE.Vector3(sx, sy, sz);

    const maxScore = coFiringList.length > 0 ? coFiringList[0].score : 1;
    const maxArcs = Math.min(MAX_SYNAPSE_ARCS, coFiringList.length);

    for (let i = 0; i < maxArcs; i++) {
        const cf = coFiringList[i];
        if (cf.id < 0 || cf.id >= NEURON_COUNT) continue;
        const tx = positions[cf.id * 3];
        const ty = positions[cf.id * 3 + 1];
        const tz = positions[cf.id * 3 + 2];
        const targetVec = new THREE.Vector3(tx, ty, tz);

        const arc = createGeodesicArc(sourceVec, targetVec, cf.score, maxScore);
        synapseGroup.add(arc);
    }
}

function showNeuronProfile() {
    document.getElementById('neuron-panel-empty').classList.add('hidden');
    document.getElementById('neuron-profile').classList.remove('hidden');
}

function renderNeuronProfile(data) {
    showNeuronProfile();

    document.getElementById('np-id').textContent = `#${data.neuron_id}`;
    const currentAct = currentActivation[data.neuron_id] || 0;
    const statusEl = document.getElementById('np-status');
    if (currentAct > 0.3) {
        statusEl.textContent = 'Firing';
        statusEl.className = 'np-status-badge firing';
    } else if (currentAct > 0.01) {
        statusEl.textContent = 'Active';
        statusEl.className = 'np-status-badge active';
    } else {
        statusEl.textContent = 'Dormant';
        statusEl.className = 'np-status-badge dormant';
    }

    document.getElementById('np-concept').textContent = data.concept_label || '—';

    // Top words
    const topWordsEl = document.getElementById('np-top-words');
    if (data.top_words && data.top_words.length > 0) {
        const maxScore = data.top_words[0].score || 1;
        topWordsEl.innerHTML = data.top_words.slice(0, 10).map(w => {
            const pct = Math.min(100, (w.score / maxScore) * 100);
            return `<div class="np-word-bar-row">
                <span class="np-word-text">${escapeHtml(w.word)}</span>
                <div class="np-word-bar-bg"><div class="np-word-bar-fill" style="width:${pct}%"></div></div>
                <span class="np-word-score">${w.score.toFixed(3)}</span>
            </div>`;
        }).join('');
    } else {
        topWordsEl.innerHTML = '<span class="np-empty-hint">No data yet</span>';
    }

    // Timeline sparkline
    lastProfileTimeline = data.activation_timeline || [];
    renderSparkline(lastProfileTimeline, data.token_chars || []);

    // Word contributions
    const contribEl = document.getElementById('np-word-contrib');
    if (data.word_contributions && data.word_contributions.length > 0) {
        const maxC = data.word_contributions[0].score || 1;
        contribEl.innerHTML = data.word_contributions.slice(0, 10).map(w => {
            const pct = Math.min(100, (w.score / maxC) * 100);
            const blocks = Math.round(pct / 10);
            const bar = '\u2588'.repeat(blocks) + '\u2591'.repeat(10 - blocks);
            return `<div class="np-word-bar-row">
                <span class="np-word-text">${escapeHtml(w.word)}</span>
                <span class="np-block-bar">${bar}</span>
                <span class="np-word-score">${w.score.toFixed(3)}</span>
            </div>`;
        }).join('');
    } else {
        contribEl.innerHTML = '<span class="np-empty-hint">No data yet</span>';
    }

    // Co-firing
    const cofireEl = document.getElementById('np-co-firing');
    if (data.co_firing && data.co_firing.length > 0) {
        cofireEl.innerHTML = data.co_firing.slice(0, 12).map(cf =>
            `<button class="np-cofire-chip" data-neuron="${cf.id}">
                #${cf.id} <span class="np-cofire-score">${cf.score.toFixed(2)}</span>
            </button>`
        ).join('');
        cofireEl.querySelectorAll('.np-cofire-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const nid = parseInt(chip.dataset.neuron);
                selectedNeuronId = nid;
                requestNeuronProfile(nid);
            });
        });
        // Draw synapse lines from this neuron to co-firing neurons
        updateSynapseLines(data.neuron_id, data.co_firing);
    } else {
        cofireEl.innerHTML = '<span class="np-empty-hint">No data yet</span>';
        clearSynapseLines();
    }
}

function renderSparkline(timeline, chars) {
    const canvas = document.getElementById('np-timeline-canvas');
    const scrollContainer = document.getElementById('np-timeline-scroll');
    const H = 80;

    if (!timeline || timeline.length < 2) {
        canvas.width = 340;
        canvas.height = H;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, 340, H);
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.font = '11px Inter';
        ctx.fillText('No timeline data yet', 340 / 2 - 55, H / 2);
        return;
    }

    // Dynamic width: 4px per token, minimum 340px
    const PX_PER_TOKEN = 4;
    const W = Math.max(340, timeline.length * PX_PER_TOKEN);
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const maxVal = Math.max(...timeline, 0.01);
    const stepX = W / (timeline.length - 1);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let y = 0; y < H; y += H / 4) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    const gradient = ctx.createLinearGradient(0, 0, 0, H);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.3)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.02)');

    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < timeline.length; i++) {
        const x = i * stepX;
        const y = H - (timeline[i] / maxVal) * (H - 8);
        ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < timeline.length; i++) {
        const x = i * stepX;
        const y = H - (timeline[i] / maxVal) * (H - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Yellow peak at current display index
    if (currentDisplayIndex >= 0 && currentDisplayIndex < timeline.length) {
        const cx = currentDisplayIndex * stepX;
        const cy = H - (timeline[currentDisplayIndex] / maxVal) * (H - 8);
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#fdd835';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Auto-scroll to keep yellow peak visible
        if (scrollContainer) {
            const containerWidth = scrollContainer.clientWidth;
            const peakX = cx;
            if (peakX < scrollContainer.scrollLeft || peakX > scrollContainer.scrollLeft + containerWidth - 20) {
                scrollContainer.scrollLeft = Math.max(0, peakX - containerWidth / 2);
            }
        }
    }
}

function updateLiveMonitor() {
    if (selectedNeuronId === null) return;
    const act = currentActivation[selectedNeuronId] || 0;
    const bar = document.getElementById('np-live-bar');
    const val = document.getElementById('np-live-value');
    if (bar) bar.style.width = `${Math.min(100, act * 100)}%`;
    if (val) val.textContent = act.toFixed(4);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ═══════════════════════════════════════════
// PLAYBACK CONTROLS
// ═══════════════════════════════════════════
const playBtn = document.getElementById('play-btn');
const pauseBtn = document.getElementById('pause-btn');
const stepBackBtn = document.getElementById('step-back-btn');
const stepFwdBtn = document.getElementById('step-fwd-btn');
const wordBackBtn = document.getElementById('word-back-btn');
const wordFwdBtn = document.getElementById('word-fwd-btn');
const playbackLabel = document.getElementById('playback-label');
const timelineSlider = document.getElementById('timeline-slider');

function updatePlaybackLabel() {
    const total = xyTimeline.length;
    const cur = currentDisplayIndex + 1;
    const ch = (currentDisplayIndex >= 0 && currentDisplayIndex < tokenChars.length)
        ? tokenChars[currentDisplayIndex] : '';
    const display = ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch;
    playbackLabel.textContent = `Token ${cur} / ${total}${display ? ` "${display}"` : ''}`;
}

playBtn.addEventListener('click', () => {
    playbackMode = 'live';
    playBtn.classList.add('active');
    pauseBtn.classList.remove('active');
    if (xyTimeline.length > 0) {
        setGlobeFromTimeline(xyTimeline.length - 1);
        timelineSlider.value = xyTimeline.length - 1;
    }
});

pauseBtn.addEventListener('click', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
});

stepBackBtn.addEventListener('click', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
    const newIdx = Math.max(0, currentDisplayIndex - 1);
    setGlobeFromTimeline(newIdx);
    timelineSlider.value = newIdx;
});

stepFwdBtn.addEventListener('click', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
    const newIdx = Math.min(xyTimeline.length - 1, currentDisplayIndex + 1);
    setGlobeFromTimeline(newIdx);
    timelineSlider.value = newIdx;
});

wordBackBtn.addEventListener('click', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
    let idx = currentDisplayIndex - 1;
    while (idx > 0 && tokenChars[idx] !== ' ') idx--;
    if (idx < 0) idx = 0;
    setGlobeFromTimeline(idx);
    timelineSlider.value = idx;
});

wordFwdBtn.addEventListener('click', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
    let idx = currentDisplayIndex + 1;
    while (idx < tokenChars.length - 1 && tokenChars[idx] !== ' ') idx++;
    if (idx >= tokenChars.length) idx = tokenChars.length - 1;
    setGlobeFromTimeline(idx);
    timelineSlider.value = idx;
});

timelineSlider.addEventListener('input', () => {
    playbackMode = 'paused';
    pauseBtn.classList.add('active');
    playBtn.classList.remove('active');
    const idx = parseInt(timelineSlider.value);
    setGlobeFromTimeline(idx);
});

// ═══════════════════════════════════════════
// UI EVENTS — Chat
// ═══════════════════════════════════════════
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');

function sendMessage() {
    const text = input.value.trim();
    if (text && socket && socket.readyState === WebSocket.OPEN && !isGenerating) {
        clearWelcome();
        currentAiBubble = null;
        isGenerating = true;
        playbackMode = 'live';
        playBtn.classList.add('active');
        pauseBtn.classList.remove('active');

        const userMsg = createUserMessage(text);
        chatHistory.appendChild(userMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        socket.send(JSON.stringify({ type: "prompt", text: text }));
        input.value = '';
    }
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

resetBtn.addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "reset" }));
        resetChatUI();
        xyTimeline = [];
        tokenChars = [];
        wordSpans = [];
        currentDisplayIndex = -1;
        targetActivation.fill(0);
        currentActivation.fill(0);
        selectedNeuronId = null;
        ablatedSet.clear();
        hideNeuronProfile();
        timelineSlider.value = 0;
        timelineSlider.max = 0;
        updatePlaybackLabel();
        updateAblationDisplay();
    }
});

function bindChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt) {
                input.value = prompt;
                sendMessage();
            }
        });
    });
}
bindChips();

// ═══════════════════════════════════════════
// CONTROLS — Hebbian, Ablation, Consolidation, Experiment B
// ═══════════════════════════════════════════

// --- Temperature / Top-K ---
const sliderTemp = document.getElementById('slider-temp');
const valTemp = document.getElementById('val-temp');
const sliderTopk = document.getElementById('slider-topk');
const valTopk = document.getElementById('val-topk');

sliderTemp.addEventListener('input', () => {
    valTemp.textContent = parseFloat(sliderTemp.value).toFixed(2);
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "set_temperature", value: parseFloat(sliderTemp.value) }));
});
sliderTopk.addEventListener('input', () => {
    valTopk.textContent = sliderTopk.value;
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "set_topk", value: parseInt(sliderTopk.value) }));
});

// --- Hebbian ---
const chkHebb = document.getElementById('chk-hebb');
const chkPersist = document.getElementById('chk-hebb-persist');
const btnResetHebb = document.getElementById('btn-reset-hebb');

chkHebb.addEventListener('change', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        // Single checkbox sends both learn and apply together
        socket.send(JSON.stringify({ type: "hebb_learn", value: chkHebb.checked }));
        socket.send(JSON.stringify({ type: "hebb_apply", value: chkHebb.checked }));
    }
});
chkPersist.addEventListener('change', () => {
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "hebb_persist", value: chkPersist.checked }));
});
btnResetHebb.addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "reset_hebb" }));
});

function updateHebbStats(maxVal, meanVal) {
    const el = document.getElementById('hebb-stats');
    if (el) el.textContent = `Memory: max=${maxVal.toFixed(4)}, mean=${meanVal.toFixed(4)}`;
}

// --- Activation Threshold ---
const sliderThreshold = document.getElementById('slider-threshold');
const valThreshold = document.getElementById('val-threshold');
sliderThreshold.addEventListener('input', () => {
    valThreshold.textContent = parseFloat(sliderThreshold.value).toFixed(2);
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "threshold", value: parseFloat(sliderThreshold.value) }));
});

// --- Hebbian Trace ---
const chkTrace = document.getElementById('chk-trace');
chkTrace.addEventListener('change', () => {
    traceEnabled = chkTrace.checked;
});

// --- Ablation ---
function updateAblationDisplay() {
    const el = document.getElementById('ablation-list');
    if (ablatedSet.size === 0) {
        el.textContent = 'No neurons ablated';
    } else {
        const list = [...ablatedSet].sort((a, b) => a - b).slice(0, 20);
        el.textContent = `Ablated (${ablatedSet.size}): ${list.map(i => '#' + i).join(', ')}${ablatedSet.size > 20 ? '...' : ''}`;
    }
}

document.getElementById('btn-clear-ablation').addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "clear_ablation" }));
});

// --- Consolidation ---
document.getElementById('btn-consolidate').addEventListener('click', async () => {
    const statsEl = document.getElementById('consolidation-stats');
    statsEl.textContent = 'Consolidating...';
    try {
        const res = await fetch('/consolidate', { method: 'POST' });
        const json = await res.json();
        if (json.status === 'success') {
            statsEl.innerHTML = `✅ Active: ${(json.active_fraction * 100).toFixed(1)}% | ` +
                `Tokens: ${json.accumulated_tokens} | ` +
                `ΔEnc: ${json.weight_norm_change?.encoder?.toFixed(4) || '?'} | ` +
                `ΔDec: ${json.weight_norm_change?.decoder?.toFixed(4) || '?'}`;
        } else {
            statsEl.textContent = `⏭ ${json.reason || json.status}`;
        }
    } catch (e) {
        statsEl.textContent = `❌ Error: ${e.message}`;
    }
});

// --- Experiment B ---
const sliderReps = document.getElementById('slider-reps');
const valReps = document.getElementById('val-reps');
sliderReps.addEventListener('input', () => { valReps.textContent = sliderReps.value; });

const btnExpB = document.getElementById('btn-exp-b');
btnExpB.addEventListener('click', () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    btnExpB.disabled = true;
    btnExpB.innerText = 'Running...';
    const expResults = document.getElementById('exp-b-results');
    expResults.textContent = 'Starting experiment...';

    const prompt = document.getElementById('exp-query').value || 'A = ';
    const teach = document.getElementById('exp-teach').value + '\n' || 'A = 1 B = 2 C = 3\n';
    const reps = parseInt(sliderReps.value) || 50;

    socket.send(JSON.stringify({ type: "experiment_b", prompt, teach, reps }));
    setTimeout(() => { btnExpB.disabled = false; btnExpB.innerText = 'Run Experiment B'; }, 60000);
});

function renderExperimentProgress(data) {
    const el = document.getElementById('exp-b-results');
    if (data.type === 'exp_phase') {
        el.textContent = `Phase: ${data.phase}...`;
    } else if (data.type === 'exp_progress') {
        if (data.phase === 'teaching') {
            el.textContent = `Teaching: ${data.rep}/${data.total} (mem_max=${data.mem_max})`;
        } else if (data.phase === 'baseline') {
            el.textContent = `Baseline: "${data.text}"`;
        }
    }
}

function renderExperimentResult(data) {
    const el = document.getElementById('exp-b-results');
    btnExpB.disabled = false;
    btnExpB.innerText = 'Run Experiment B';
    el.innerHTML = `
        <div class="exp-result-item"><strong>Prompt:</strong> <code>${escapeHtml(data.prompt)}</code></div>
        <div class="exp-result-item"><strong>Baseline:</strong> ${escapeHtml(data.baseline_text)}</div>
        <div class="exp-result-item"><strong>After Hebb:</strong> ${escapeHtml(data.test_text)}</div>
        <div class="exp-result-item"><strong>Cosine Sim:</strong> ${data.cosine_similarity}</div>
        <div class="exp-result-item"><strong>Changed Neurons:</strong> ${data.changed_neurons}</div>
        <div class="exp-result-item"><strong>Memory:</strong> max=${data.mem_max}, sparsity=${data.mem_sparsity}%</div>
    `;
}

// ═══════════════════════════════════════════
// ANIMATION LOOP
// ═══════════════════════════════════════════
const fpsText = document.getElementById('fps-counter');
let lastFrameTime = performance.now();
let frames = 0;

function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    frames++;
    if (now > lastFrameTime + 1000) {
        const fps = Math.round((frames * 1000) / (now - lastFrameTime));
        if (socket && socket.readyState === WebSocket.OPEN) {
            fpsText.innerText = `${fps} FPS`;
        }
        lastFrameTime = now;
        frames = 0;
    }

    controls.update();
    updateGlobeColors();
    updateLiveMonitor();

    // Synapse line breathing pulse
    if (synapseGroup.children.length > 0 && selectedNeuronId !== null) {
        const time = performance.now() * 0.001;
        const pulse = 0.85 + Math.sin(time * 2.5) * 0.15; // 0.7 to 1.0 multiplier
        synapseGroup.children.forEach(line => {
            line.material.opacity = line.userData.baseOpacity * pulse;
        });
    }

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0) {
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
});

connect();
animate();

// ═══════════════════════════════════════════
// VIZ CONTROLS (zoom, rotate, views)
// ═══════════════════════════════════════════
const zoomSlider = document.getElementById('zoom-slider');
const zoomInBtn = document.getElementById('zoom-in-btn');
const zoomOutBtn = document.getElementById('zoom-out-btn');
const autoRotateBtn = document.getElementById('auto-rotate-btn');
const resetViewBtn = document.getElementById('reset-view-btn');
const topViewBtn = document.getElementById('top-view-btn');
const sideViewBtn = document.getElementById('side-view-btn');

zoomSlider.addEventListener('input', () => {
    const dist = parseFloat(zoomSlider.value);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
});

zoomInBtn.addEventListener('click', () => {
    const dist = Math.max(8, camera.position.length() - 5);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
    zoomSlider.value = dist;
});

zoomOutBtn.addEventListener('click', () => {
    const dist = Math.min(80, camera.position.length() + 5);
    const dir = camera.position.clone().normalize();
    camera.position.copy(dir.multiplyScalar(dist));
    zoomSlider.value = dist;
});

const freezeBtn = document.getElementById('freeze-btn');
let isFrozen = false;
let wasAutoRotating = false;

const PAUSE_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
const PLAY_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;

freezeBtn.addEventListener('click', () => {
    isFrozen = !isFrozen;
    if (isFrozen) {
        wasAutoRotating = controls.autoRotate;
        controls.autoRotate = false;
        // Keep manual drag/rotate/zoom/pan enabled!
        freezeBtn.classList.add('active');
        freezeBtn.innerHTML = PLAY_ICON;
        autoRotateBtn.classList.remove('active');
    } else {
        controls.autoRotate = wasAutoRotating;
        autoRotateBtn.classList.toggle('active', wasAutoRotating);
        freezeBtn.classList.remove('active');
        freezeBtn.innerHTML = PAUSE_ICON;
    }
});

autoRotateBtn.addEventListener('click', () => {
    if (isFrozen) return;
    controls.autoRotate = !controls.autoRotate;
    autoRotateBtn.classList.toggle('active', controls.autoRotate);
});

resetViewBtn.addEventListener('click', () => {
    if (isFrozen) {
        isFrozen = false;
        controls.enableRotate = true;
        controls.enableZoom = true;
        controls.enablePan = true;
        freezeBtn.classList.remove('active');
        freezeBtn.innerHTML = PAUSE_ICON;
    }
    camera.position.set(0, 0, 25);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    zoomSlider.value = 25;
});

topViewBtn.addEventListener('click', () => {
    const dist = camera.position.length();
    camera.position.set(0, dist, 0);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
});

sideViewBtn.addEventListener('click', () => {
    const dist = camera.position.length();
    camera.position.set(dist, 0, 0);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
});
