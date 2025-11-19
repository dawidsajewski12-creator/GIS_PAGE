/**
 * GIS Microclimate Platform v3.0 – App.js PRODUCTION
 * 
 * FIXED & WORKING:
 * - API paths corrected to work with GitHub Pages structure
 * - Data loading with debugging
 * - Scientific visualization with real results
 * - Results display panel with statistics
 * - Professional styling
 */

const CONFIG = {
  // API paths - works with GitHub Pages
  API_BASE: 'https://raw.githubusercontent.com/dawidsajewski12-creator/GIS_PAGE/main/api/data',
  WIND_DATA_URL: 'https://raw.githubusercontent.com/dawidsajewski12-creator/GIS_PAGE/main/api/data/wind_simulation/current.json',
  
  // Or use relative if serving locally:
  // API_BASE: './api/data',
  // WIND_DATA_URL: './api/data/wind_simulation/current.json',
  
  // Map defaults (Warszawa)
  DEFAULT_LAT: 52.2297,
  DEFAULT_LNG: 21.0122,
  DEFAULT_ZOOM: 13,
  
  // Visualization
  PARTICLE_COUNT: 800,
  PARTICLE_LIFESPAN: 600,
  STREAMLINE_COUNT: 200,
  HEATMAP_COLORMAP: 'jet',
};

let STATE = {
  map: null,
  layers: { heatmap: null, streamlines: null, particles: null },
  data: { windSimulation: null, lastUpdate: null },
  ui: { isDarkMode: false },
};

// ============================================================================
// UTILITIES
// ============================================================================

function getColor(value, colormap = 'jet') {
  const t = Math.max(0, Math.min(1, value));
  let r, g, b;
  
  if (colormap === 'jet') {
    if (t < 0.125) {
      r = 0; g = 0; b = Math.round(255 * (0.5 + t * 4));
    } else if (t < 0.375) {
      r = 0; g = Math.round(255 * ((t - 0.125) * 4)); b = 255;
    } else if (t < 0.625) {
      r = Math.round(255 * ((t - 0.375) * 4)); g = 255; b = Math.round(255 * (1 - (t - 0.375) * 4));
    } else if (t < 0.875) {
      r = 255; g = Math.round(255 * (1 - (t - 0.625) * 4)); b = 0;
    } else {
      r = Math.round(255 * (1 - (t - 0.875) * 2)); g = 0; b = 0;
    }
  } else {
    r = Math.round(255 * t);
    g = Math.round(255 * (1 - Math.abs(t - 0.5)));
    b = Math.round(255 * (1 - t));
  }
  
  return { r, g, b, a: 0.7 };
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadWindData() {
  console.log(`📥 Attempting to load: ${CONFIG.WIND_DATA_URL}`);
  
  try {
    const response = await fetch(CONFIG.WIND_DATA_URL);
    console.log(`Response status: ${response.status}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ Data loaded:', {
      vectors: data.vector_field?.length || 0,
      streamlines: data.streamlines?.length || 0,
      particles: data.particles?.length || 0,
      stats: data.flow_statistics,
    });
    
    STATE.data.windSimulation = data;
    STATE.data.lastUpdate = new Date().toISOString();
    
    return data;
  } catch (error) {
    console.error('❌ Data loading failed:', error);
    showError(`Failed to load data: ${error.message}`);
    return null;
  }
}

function showError(message) {
  const container = document.getElementById('wind-map');
  if (container) {
    container.innerHTML = `
      <div style="padding: 20px; background: #fee; color: #c00; border-radius: 8px; margin: 10px;">
        <strong>⚠️ Error:</strong> ${message}
        <br><br>
        <small>Check console (F12) for details</small>
      </div>
    `;
  }
}

// ============================================================================
// VISUALIZATION LAYERS
// ============================================================================

class HeatmapLayer {
  constructor(data, bounds) {
    this.data = data;
    this.bounds = bounds;
  }
  
  addTo(map) {
    this.map = map;
    this.canvas = L.DomUtil.create('canvas', 'heatmap-canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    
    map.on('moveend zoomend resize', () => this._reset(), this);
    this._reset();
    return this;
  }
  
  remove() {
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
  }
  
  _reset() {
    if (!this.map) return;
    const topLeft = this.map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this.canvas, topLeft);
    const size = this.map.getSize();
    this.canvas.width = size.x;
    this.canvas.height = size.y;
    this._draw();
  }
  
  _draw() {
    if (!this.data.vector_field || this.data.vector_field.length === 0) return;
    
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    const stats = this.data.flow_statistics || {};
    const minMag = stats.min_magnitude || 0;
    const maxMag = stats.max_magnitude || 1;
    
    // Draw velocity vectors as circles
    this.data.vector_field.forEach(v => {
      const lat = v.latitude || v.lat;
      const lng = v.longitude || v.lng;
      
      if (!lat || !lng) return;
      
      const point = this.map.latLngToContainerPoint([lat, lng]);
      const magnitude = v.magnitude || 0;
      const normalized = (magnitude - minMag) / (maxMag - minMag || 1);
      const color = getColor(normalized, CONFIG.HEATMAP_COLORMAP);
      
      this.ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`;
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
      this.ctx.fill();
    });
  }
}

class StreamlineLayer {
  constructor(data, bounds) {
    this.data = data;
    this.bounds = bounds;
  }
  
  addTo(map) {
    this.map = map;
    this.canvas = L.DomUtil.create('canvas', 'streamline-canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    
    map.on('moveend zoomend resize', () => this._reset(), this);
    this._reset();
    return this;
  }
  
  remove() {
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
  }
  
  _reset() {
    if (!this.map) return;
    const topLeft = this.map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this.canvas, topLeft);
    const size = this.map.getSize();
    this.canvas.width = size.x;
    this.canvas.height = size.y;
    this._draw();
  }
  
  _draw() {
    if (!this.data.streamlines) return;
    
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = 'rgba(100, 160, 200, 0.7)';
    this.ctx.lineWidth = 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    
    this.data.streamlines.forEach(streamline => {
      if (streamline.length < 2) return;
      
      this.ctx.beginPath();
      
      streamline.forEach((point, idx) => {
        const lat = point.latitude || point.lat || point.y;
        const lng = point.longitude || point.lng || point.x;
        
        if (lat === undefined || lng === undefined) return;
        
        const containerPoint = this.map.latLngToContainerPoint([lat, lng]);
        
        if (idx === 0) {
          this.ctx.moveTo(containerPoint.x, containerPoint.y);
        } else {
          this.ctx.lineTo(containerPoint.x, containerPoint.y);
        }
      });
      
      this.ctx.stroke();
    });
  }
}

// ============================================================================
// MAP INITIALIZATION
// ============================================================================

async function initializeMap() {
  console.log('🗺️ Initializing map...');
  
  const mapContainer = document.getElementById('wind-map');
  if (!mapContainer) {
    console.error('❌ Map container not found');
    return;
  }
  
  // Create map
  STATE.map = L.map(mapContainer).setView([CONFIG.DEFAULT_LAT, CONFIG.DEFAULT_LNG], CONFIG.DEFAULT_ZOOM);
  
  // Add basemap
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    opacity: 0.8,
  }).addTo(STATE.map);
  
  // Load wind data
  const windData = await loadWindData();
  if (!windData) {
    showError('Could not load wind simulation data');
    return;
  }
  
  // Calculate bounds from data
  const lats = windData.vector_field
    .map(v => v.latitude || v.lat)
    .filter(Boolean);
  const lngs = windData.vector_field
    .map(v => v.longitude || v.lng)
    .filter(Boolean);
  
  if (lats.length === 0 || lngs.length === 0) {
    console.warn('⚠️ No geographic coordinates found, using default bounds');
    const bounds = L.latLngBounds([[52.0, 20.8], [52.4, 21.4]]);
    STATE.map.fitBounds(bounds);
  } else {
    const bounds = L.latLngBounds(
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)]
    );
    STATE.map.fitBounds(bounds);
  }
  
  // Add visualization layers
  STATE.layers.heatmap = new HeatmapLayer(windData, null).addTo(STATE.map);
  STATE.layers.streamlines = new StreamlineLayer(windData, null).addTo(STATE.map);
  
  console.log('✅ Map initialized');
  updateResultsPanel(windData);
}

// ============================================================================
// RESULTS DISPLAY
// ============================================================================

function updateResultsPanel(data) {
  const stats = data.flow_statistics || {};
  const perf = data.performance || {};
  
  const panel = document.getElementById('results-panel');
  if (!panel) return;
  
  const html = `
    <div class="results-header">
      <h2>Wyniki Symulacji Przepływu Wiatru</h2>
      <p class="subtitle">CFD - Metoda Lattice Boltzmann</p>
    </div>
    
    <div class="results-grid">
      <div class="result-card">
        <div class="result-icon">🌪️</div>
        <div class="result-content">
          <div class="result-label">Maksymalna Prędkość</div>
          <div class="result-value">${(stats.max_magnitude || 0).toFixed(2)} <span class="unit">m/s</span></div>
        </div>
      </div>
      
      <div class="result-card">
        <div class="result-icon">📊</div>
        <div class="result-content">
          <div class="result-label">Średnia Prędkość</div>
          <div class="result-value">${(stats.mean_magnitude || 0).toFixed(2)} <span class="unit">m/s</span></div>
        </div>
      </div>
      
      <div class="result-card">
        <div class="result-icon">⚡</div>
        <div class="result-content">
          <div class="result-label">Intensywność Turbulencji</div>
          <div class="result-value">${(stats.turbulence_intensity || 0).toFixed(3)}</div>
        </div>
      </div>
      
      <div class="result-card">
        <div class="result-icon">⏱️</div>
        <div class="result-content">
          <div class="result-label">Czas Obliczenia</div>
          <div class="result-value">${(perf.simulation_time || 0).toFixed(1)} <span class="unit">s</span></div>
        </div>
      </div>
      
      <div class="result-card">
        <div class="result-icon">🔢</div>
        <div class="result-content">
          <div class="result-label">Liczba Wektorów</div>
          <div class="result-value">${(data.vector_field || []).length.toLocaleString()}</div>
        </div>
      </div>
      
      <div class="result-card">
        <div class="result-icon">〰️</div>
        <div class="result-content">
          <div class="result-label">Linie Prądu</div>
          <div class="result-value">${(data.streamlines || []).length}</div>
        </div>
      </div>
    </div>
    
    <div class="results-info">
      <p>
        <strong>Metodologia:</strong> Symulacja oparta na metodzie Lattice Boltzmann (D2Q9) 
        z równaniem BGK dla D2Q9 sieci. Obliczenia wykonane na Google Colab z użyciem 
        optymalizacji Numba JIT.
      </p>
      <p>
        <strong>Dane wejściowe:</strong> Numeryczny Model Terenu (NMT), Numeryczny Model 
        Powierzchni (NMPT), geometria budynków z OpenStreetMap.
      </p>
      <p>
        <strong>Wyniki:</strong> Pole wektorowe prędkości wiatru, linie prądu (streamlines), 
        dane o cząsteczkach dla wizualizacji dynamiki przepływu.
      </p>
    </div>
  `;
  
  panel.innerHTML = html;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Initializing GIS Microclimate Platform...');
  initializeMap().catch(err => {
    console.error('Initialization failed:', err);
    showError('Initialization failed: ' + err.message);
  });
});

// Export for debugging
window.STATE = STATE;
window.loadWindData = loadWindData;
window.initializeMap = initializeMap;
