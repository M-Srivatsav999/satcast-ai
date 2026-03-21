// ══════════════════════════════════════════════════════
// SatCast AI — Main JavaScript
// ══════════════════════════════════════════════════════

// ── State ─────────────────────────────────────────────
let currentFrames = []; // base64 frames
let currentMode = 'nasa';
let currentResult = null;
let forecastChart = null;
let gaugeChart = null;
let probChart = null;
let weatherChart = null;

// ── Init ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initParticles();
    initNavigation();
    initControls();
    initDragDrop();
});

// ── Clock ─────────────────────────────────────────────
function initClock() {
    function tick() {
        const now = new Date();
        const t = now.toUTCString().split(' ')[4];
        const el = document.getElementById('navTime');
        if (el) el.textContent = t + ' UTC';
    }
    tick();
    setInterval(tick, 1000);
}

// ── Rain Animation ────────────────────────────────────
function initParticles() {
    const canvas = document.getElementById('rainCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H;
    const drops = [];

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 120; i++) {
        drops.push({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            len: Math.random() * 18 + 6,
            speed: Math.random() * 3 + 2,
            alpha: Math.random() * 0.4 + 0.05,
            width: Math.random() * 0.8 + 0.3,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        drops.forEach(d => {
            ctx.beginPath();
            ctx.moveTo(d.x, d.y);
            ctx.lineTo(d.x - 1, d.y + d.len);
            ctx.strokeStyle = `rgba(0,168,232,${d.alpha})`;
            ctx.lineWidth = d.width;
            ctx.stroke();
            d.y += d.speed;
            d.x -= 0.5;
            if (d.y > H + 20) {
                d.y = -20;
                d.x = Math.random() * W;
            }
        });
        requestAnimationFrame(draw);
    }
    draw();
}

// ── Navigation ────────────────────────────────────────
function initNavigation() {
    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const tab = link.dataset.tab;
            switchTab(tab);
            links.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function switchTab(tab) {
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById('tab-' + tab);
    if (target) target.classList.add('active');
}

// ── Controls ──────────────────────────────────────────
function initControls() {
    const loc = document.getElementById('locationSelect');
    if (loc) loc.addEventListener('change', () => {
        const custom = document.getElementById('customCoords');
        if (custom) custom.style.display = loc.value === 'Custom (enter coords)' ? 'block' : 'none';
    });

    const frames = document.getElementById('framesSlider');
    const framesVal = document.getElementById('framesVal');
    if (frames) frames.addEventListener('input', () => { if (framesVal) framesVal.textContent = frames.value; });

    const margin = document.getElementById('customMargin');
    const marginVal = document.getElementById('marginVal');
    if (margin) margin.addEventListener('input', () => { if (marginVal) marginVal.textContent = parseFloat(margin.value).toFixed(1); });
}

function setMode(mode) {
    currentMode = mode;
    document.getElementById('modeNasa').classList.toggle('active', mode === 'nasa');
    document.getElementById('modeUpload').classList.toggle('active', mode === 'upload');
    document.getElementById('uploadSection').style.display = mode === 'upload' ? 'block' : 'none';
    document.getElementById('fetchBtn').style.display = mode === 'nasa' ? 'inline-flex' : 'none';
    currentFrames = [];
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('framesGrid').style.display = 'none';
    document.getElementById('framesPlaceholder').style.display = 'block';
}

// ── Drag & Drop ───────────────────────────────────────
function initDragDrop() {
    const zone = document.getElementById('uploadZone');
    if (!zone) return;
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.style.borderColor = '#00d4ff';
    });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = '';
        handleFiles(e.dataTransfer.files);
    });
}

function handleUpload(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    currentFrames = [];
    const promises = Array.from(files).map(file => {
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    });

    Promise.all(promises).then(results => {
        currentFrames = results;
        displayFrames(results.map((b64, i) => ({ b64: b64.split(',')[1], date: `Frame ${i+1}` })));
        document.getElementById('predictBtn').disabled = false;
    });
}

// ── Loading ───────────────────────────────────────────
function showLoading(text = 'Processing...') {
    const ol = document.getElementById('loadingOverlay');
    const lt = document.getElementById('loadingText');
    if (ol) { ol.classList.add('active'); if (lt) lt.textContent = text; }
}

function hideLoading() {
    const ol = document.getElementById('loadingOverlay');
    if (ol) ol.classList.remove('active');
}

// ── Fetch Satellite ───────────────────────────────────
async function fetchSatellite() {
    showLoading('Scanning Satellite Imagery...');
    try {
        const location = document.getElementById('locationSelect').value;
        const layer = document.getElementById('layerSelect').value;
        const n_frames = parseInt(document.getElementById('framesSlider').value);

        let bbox = null;
        if (location === 'Custom (enter coords)') {
            const lat = parseFloat(document.getElementById('customLat').value);
            const lon = parseFloat(document.getElementById('customLon').value);
            const m = parseFloat(document.getElementById('customMargin').value);
            bbox = [lon - m, lat - m, lon + m, lat + m];
        }

        const res = await fetch('/api/fetch-satellite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location, layer, n_frames, bbox })
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        currentFrames = data.frames.map(f => 'data:image/png;base64,' + f.b64);
        displayFrames(data.frames);
        document.getElementById('predictBtn').disabled = false;

        showToast(`✅ ${data.count} satellite frames loaded`, 'success');
    } catch (err) {
        showToast('❌ ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Display Frames ────────────────────────────────────
function displayFrames(frames) {
    const grid = document.getElementById('framesGrid');
    const ph = document.getElementById('framesPlaceholder');
    if (!grid) return;

    grid.style.gridTemplateColumns = `repeat(${frames.length}, 1fr)`;
    grid.innerHTML = frames.map(f => `
    <div class="frame-item">
      <img src="data:image/png;base64,${f.b64}" class="frame-img" alt="Satellite frame"/>
      <div class="frame-date">${f.date || ''}</div>
    </div>
  `).join('');

    grid.style.display = 'grid';
    ph.style.display = 'none';
}

// ── Run Prediction ────────────────────────────────────
async function runPredict() {
    if (!currentFrames.length) return;
    showLoading('Analysing Cloud Patterns...');

    try {
        const location = document.getElementById('locationSelect').value;
        const frames = currentFrames.map(f => typeof f === 'string' ? f : f);

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: currentMode, frames, location })
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        currentResult = data;
        renderResults(data);
        showToast('⚡ Prediction complete', 'success');
    } catch (err) {
        showToast('❌ ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Render Results ────────────────────────────────────
function renderResults(data) {
    document.getElementById('resultsSection').style.display = 'block';

    // Alert
    const alertEl = document.getElementById('alertBanner');
    if (data.primary_cat === 'Extreme') {
        alertEl.style.display = 'block';
        alertEl.innerHTML = `<div class="alert-extreme">
      <div class="alert-title">🚨 EXTREME RAINFALL ALERT</div>
      <div class="alert-body">Predicted intensity: <strong>${data.primary_mm_h} mm/h</strong> — Flash flood risk HIGH. Seek shelter immediately.</div>
    </div>`;
    } else if (data.primary_cat === 'Heavy') {
        alertEl.style.display = 'block';
        alertEl.innerHTML = `<div class="alert-heavy">
      <div class="alert-title">⚠️ HEAVY RAIN WARNING</div>
      <div class="alert-body">Predicted intensity: <strong>${data.primary_mm_h} mm/h</strong> — Waterlogging possible. Drive carefully.</div>
    </div>`;
    } else {
        alertEl.style.display = 'none';
    }

    // Metrics
    const col = data.primary_color;
    document.getElementById('metricsRow').innerHTML = `
    <div class="metric-card">
      <div class="metric-icon">🌧️</div>
      <div class="metric-value" style="color:${col}">${data.primary_mm_h}</div>
      <div class="metric-label">mm/h · 1h forecast</div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">⚠️</div>
      <div class="metric-value" style="color:${col}; font-size:20px;">${data.primary_cat}</div>
      <div class="metric-label">Severity Level</div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">🎯</div>
      <div class="metric-value">${data.confidence_pct}%</div>
      <div class="metric-label">Confidence</div>
    </div>
    <div class="metric-card">
      <div class="metric-icon">🧠</div>
      <div class="metric-value" style="font-size:16px;">BiLSTM+Attn</div>
      <div class="metric-label">Architecture</div>
    </div>
  `;

    // Confidence bar
    document.getElementById('confPct').textContent = data.confidence_pct + '%';
    const fill = document.getElementById('confFill');
    setTimeout(() => { fill.style.width = data.confidence_pct + '%'; }, 100);

    // Horizon cards
    document.getElementById('horizonGrid').innerHTML = data.horizons.map(h => `
    <div class="horizon-card">
      <div class="horizon-time">${h.hours}h</div>
      <div class="horizon-value" style="color:${h.color}">${h.mm_h}</div>
      <div class="horizon-cat" style="background:${h.color}22; color:${h.color}">${h.category}</div>
      <div class="horizon-unc">±${h.uncertainty}</div>
    </div>
  `).join('');

    // Charts
    renderForecastChart(data);
    renderGaugeChart(data.confidence_pct, col);
    renderProbChart(data.primary_mm_h);

    // Cloud score
    if (data.cloud_score !== null && data.cloud_score !== undefined) {
        document.getElementById('cloudScoreBar').style.display = 'block';
        document.getElementById('csVal').textContent = data.cloud_score.toFixed(3);
        document.getElementById('csDesc').textContent = data.cloud_score < 0.2 ? 'Low cloud density' : data.cloud_score < 0.45 ? 'Medium cloud density' : 'High cloud density';
        setTimeout(() => {
            document.getElementById('csFill').style.width = (data.cloud_score * 100) + '%';
        }, 100);
    }

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Forecast Chart ────────────────────────────────────
function renderForecastChart(data) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    if (forecastChart) forecastChart.destroy();

    const hrs = data.horizons.map(h => h.hours + 'h');
    const vals = data.horizons.map(h => h.mm_h);
    const upper = data.horizons.map(h => h.mm_h + h.uncertainty);
    const lower = data.horizons.map(h => Math.max(0, h.mm_h - h.uncertainty));
    const colors = data.horizons.map(h => h.color);

    forecastChart = new Chart(ctx, {
        data: {
            labels: hrs,
            datasets: [{
                    type: 'line',
                    label: 'Upper bound',
                    data: upper,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0,212,255,0.06)',
                    fill: '+1',
                    pointRadius: 0,
                    tension: 0.4,
                },
                {
                    type: 'line',
                    label: 'Lower bound',
                    data: lower,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0,212,255,0.06)',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4,
                },
                {
                    type: 'line',
                    label: 'Rainfall (mm/h)',
                    data: vals,
                    borderColor: '#00d4ff',
                    borderWidth: 3,
                    backgroundColor: colors,
                    pointRadius: 8,
                    pointHoverRadius: 12,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    fill: false,
                    tension: 0.4,
                },
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#080f20',
                    borderColor: '#0e2040',
                    borderWidth: 1,
                    titleColor: '#e8f4ff',
                    bodyColor: '#5a8cb8',
                    callbacks: {
                        label: ctx => ` ${ctx.parsed.y} mm/h`,
                    }
                },
                annotation: {},
            },
            scales: {
                x: {
                    grid: { color: 'rgba(14,32,64,0.8)' },
                    ticks: { color: '#5a8cb8', font: { family: 'JetBrains Mono', size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(14,32,64,0.8)' },
                    ticks: { color: '#5a8cb8', font: { family: 'JetBrains Mono', size: 11 } },
                    beginAtZero: true,
                }
            }
        }
    });
}

// ── Gauge Chart ───────────────────────────────────────
function renderGaugeChart(value, color) {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    if (gaugeChart) gaugeChart.destroy();

    const rem = 100 - value;
    gaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [value, rem],
                backgroundColor: [color, '#0e2040'],
                borderWidth: 0,
                circumference: 220,
                rotation: 250,
            }]
        },
        options: {
            responsive: true,
            cutout: '78%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            }
        },
        plugins: [{
            id: 'gauge-text',
            afterDraw(chart) {
                const { ctx, chartArea: { left, right, top, bottom } } = chart;
                const cx = (left + right) / 2,
                    cy = (top + bottom) / 2 + 10;
                ctx.save();
                ctx.textAlign = 'center';
                ctx.fillStyle = '#e8f4ff';
                ctx.font = "700 22px 'Syne'";
                ctx.fillText(value + '%', cx, cy);
                ctx.fillStyle = '#5a8cb8';
                ctx.font = "400 10px 'JetBrains Mono'";
                ctx.fillText('CONFIDENCE', cx, cy + 16);
                ctx.restore();
            }
        }]
    });
}

// ── Rain Probability Chart ────────────────────────────
function renderProbChart(mm_h) {
    const ctx = document.getElementById('probChart').getContext('2d');
    if (probChart) probChart.destroy();

    const prob = Math.min(99, mm_h < 1 ? mm_h * 30 : mm_h < 5 ? 30 + mm_h * 10 : Math.min(99, 70 + mm_h * 1.5));
    const color = prob < 30 ? '#00e676' : prob < 60 ? '#ffd740' : prob < 85 ? '#ff6d00' : '#ff1744';

    probChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [prob, 100 - prob],
                backgroundColor: [color, '#0e2040'],
                borderWidth: 0,
                circumference: 220,
                rotation: 250,
            }]
        },
        options: {
            responsive: true,
            cutout: '78%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        },
        plugins: [{
            id: 'prob-text',
            afterDraw(chart) {
                const { ctx, chartArea: { left, right, top, bottom } } = chart;
                const cx = (left + right) / 2,
                    cy = (top + bottom) / 2 + 10;
                ctx.save();
                ctx.textAlign = 'center';
                ctx.fillStyle = color;
                ctx.font = "700 22px 'Syne'";
                ctx.fillText(Math.round(prob) + '%', cx, cy);
                ctx.fillStyle = '#5a8cb8';
                ctx.font = "400 10px 'JetBrains Mono'";
                ctx.fillText('RAIN PROB', cx, cy + 16);
                ctx.restore();
            }
        }]
    });
}

// ── Satellite View ────────────────────────────────────
async function loadSatView() {
    showLoading('Loading Atmospheric Layers...');
    try {
        const checked = Array.from(document.querySelectorAll('#layerChecks input:checked')).map(c => c.value);
        if (!checked.length) {
            hideLoading();
            showToast('Select at least one layer', 'error');
            return;
        }

        const location = document.getElementById('locationSelect').value;
        const date = document.getElementById('satDate').value;
        const resolution = document.getElementById('satRes').value;

        let bbox = null;
        if (location === 'Custom (enter coords)') {
            const lat = parseFloat(document.getElementById('customLat').value);
            const lon = parseFloat(document.getElementById('customLon').value);
            const m = parseFloat(document.getElementById('customMargin').value);
            bbox = [lon - m, lat - m, lon + m, lat + m];
        }

        const res = await fetch('/api/satellite-view', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ layers: checked, location, date, resolution, bbox })
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const grid = document.getElementById('satViewGrid');
        grid.innerHTML = data.layers.map(l => l.error ?
            `<div class="sat-layer-card"><div class="sat-layer-info"><div class="sat-layer-name">${l.layer}</div><div style="color:#ff6666;font-size:12px;">${l.error}</div></div></div>` :
            `<div class="sat-layer-card">
          <img src="data:image/png;base64,${l.b64}" class="sat-layer-img" alt="${l.layer}"/>
          <div class="sat-layer-info">
            <div class="sat-layer-name">${l.layer}</div>
            <div class="sat-layer-stats">
              <div class="sat-stat"><div class="sat-stat-label">Cloud</div><div class="sat-stat-val">${l.cloud}</div></div>
              <div class="sat-stat"><div class="sat-stat-label">Cold Tops</div><div class="sat-stat-val">${l.cold}</div></div>
              <div class="sat-stat"><div class="sat-stat-label">Moisture</div><div class="sat-stat-val">${l.moisture}</div></div>
              <div class="sat-stat"><div class="sat-stat-label">Texture</div><div class="sat-stat-val">${l.texture}</div></div>
            </div>
          </div>
        </div>`
        ).join('');

        showToast(`✅ ${data.layers.length} layers loaded`, 'success');
    } catch (err) {
        showToast('❌ ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Weather ───────────────────────────────────────────
async function fetchWeather() {
    showLoading('Reading Atmospheric Conditions...');
    try {
        const location = document.getElementById('locationSelect').value;

        const coordMap = {
            'India (Hyderabad)': [17.4, 78.5],
            'India (Mumbai)': [19.1, 72.9],
            'India (Chennai)': [13.1, 80.3],
            'Bay of Bengal': [15.0, 87.5],
            'Arabian Sea': [15.0, 66.5],
            'South Asia (Full)': [22.5, 80.0],
            'Southeast Asia': [10.0, 117.5],
            'Western Pacific': [10.0, 150.0],
            'Gulf of Mexico': [25.0, -87.5],
        };

        let lat = 17.4,
            lon = 78.5;
        if (location === 'Custom (enter coords)') {
            lat = parseFloat(document.getElementById('customLat').value);
            lon = parseFloat(document.getElementById('customLon').value);
        } else if (coordMap[location]) {
            [lat, lon] = coordMap[location];
        }

        const res = await fetch('/api/weather', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon })
        });
        const wx = await res.json();
        if (wx.error) throw new Error(wx.error);

        document.getElementById('wxPlaceholder').style.display = 'none';
        document.getElementById('wxSection').style.display = 'block';

        document.getElementById('wxGrid').innerHTML = [
            ['🌧️', wx.current_rain_mm + ' mm/h', 'Current Rain'],
            ['☁️', wx.cloud_cover_pct + '%', 'Cloud Cover'],
            ['💧', wx.humidity_pct + '%', 'Humidity'],
            ['🌬️', wx.wind_speed_ms + ' m/s', 'Wind Speed'],
        ].map(([icon, val, label]) => `
      <div class="wx-card">
        <span class="wx-icon">${icon}</span>
        <div class="wx-val">${val}</div>
        <div class="wx-label">${label}</div>
      </div>
    `).join('');

        renderWeatherChart(wx.hourly_precip_24h, wx.hourly_prob_24h);

        const precip = wx.hourly_precip_24h;
        const prob = wx.hourly_prob_24h;
        document.getElementById('wxStats').innerHTML = [
            ['⬆️', 'Peak Rain', Math.max(...precip).toFixed(1) + ' mm/h'],
            ['🎯', 'Max Probability', Math.max(...prob).toFixed(0) + '%'],
            ['⏱️', 'Rainy Hours', precip.filter(p => p > 0.5).length + ' / 24'],
            ['💧', 'Total (24h)', precip.reduce((a, b) => a + b, 0).toFixed(1) + ' mm'],
        ].map(([icon, label, val]) => `
      <div class="wx-stat-pill">
        <span>${icon}</span>
        <div>
          <div class="wx-stat-label">${label}</div>
          <div class="wx-stat-val">${val}</div>
        </div>
      </div>
    `).join('');

        showToast('✅ Weather data loaded', 'success');
    } catch (err) {
        showToast('❌ ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

function renderWeatherChart(precip, prob) {
    const ctx = document.getElementById('weatherChart').getContext('2d');
    if (weatherChart) weatherChart.destroy();

    const labels = Array.from({ length: 24 }, (_, i) => i.toString().padStart(2, '0') + ':00');
    const nowHr = new Date().getUTCHours();

    weatherChart = new Chart(ctx, {
        data: {
            labels,
            datasets: [{
                    type: 'bar',
                    label: 'Precipitation (mm/h)',
                    data: precip,
                    backgroundColor: precip.map(v => v < 1 ? 'rgba(0,230,118,0.7)' : v < 5 ? 'rgba(255,215,64,0.7)' : v < 15 ? 'rgba(255,109,0,0.7)' : 'rgba(213,0,249,0.7)'),
                    borderRadius: 4,
                },
                {
                    type: 'line',
                    label: 'Probability (%)',
                    data: prob,
                    borderColor: '#00d4ff',
                    borderWidth: 2,
                    fill: true,
                    backgroundColor: 'rgba(0,212,255,0.04)',
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y2',
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#5a8cb8', font: { size: 11 } } },
                tooltip: {
                    backgroundColor: '#080f20',
                    borderColor: '#0e2040',
                    borderWidth: 1,
                    titleColor: '#e8f4ff',
                    bodyColor: '#5a8cb8',
                },
                annotation: {
                    annotations: {
                        nowLine: {
                            type: 'line',
                            xMin: nowHr,
                            xMax: nowHr,
                            borderColor: '#7b61ff',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            label: { content: 'Now', enabled: true, color: '#7b61ff', font: { size: 10 } }
                        }
                    }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(14,32,64,0.8)' }, ticks: { color: '#5a8cb8', font: { size: 10 } } },
                y: { grid: { color: 'rgba(14,32,64,0.8)' }, ticks: { color: '#5a8cb8', font: { size: 10 } }, beginAtZero: true },
                y2: { position: 'right', min: 0, max: 100, grid: { display: false }, ticks: { color: '#00d4ff', font: { size: 10 } } }
            }
        }
    });
}

// ── Download Report ───────────────────────────────────
function downloadReport() {
    if (!currentResult) return;
    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satcast_report_${new Date().toISOString().slice(0,16).replace(':','-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ── Toast Notifications ───────────────────────────────
function showToast(msg, type = 'success') {
    const toast = document.createElement('div');
    toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:${type==='error'?'rgba(255,23,68,0.15)':'rgba(0,230,118,0.12)'};
    border:1px solid ${type==='error'?'#ff1744':'#00e676'};
    color:${type==='error'?'#ff6666':'#00e676'};
    font-family:'JetBrains Mono',monospace; font-size:12px;
    padding:12px 20px; border-radius:8px;
    backdrop-filter:blur(12px);
    animation:slideIn 0.3s ease;
  `;
    toast.textContent = msg;

    const style = document.createElement('style');
    style.textContent = '@keyframes slideIn{from{transform:translateX(20px);opacity:0}to{transform:translateX(0);opacity:1}}';
    document.head.appendChild(style);

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}