/**
 * GIS Microclimate Platform v3.0 – App.js FIXED
 * Fixed version based on actual code analysis
 * 
 * Changes:
 * - FIXED: API data path to use relative paths (./api/...)
 * - FIXED: Leaflet layer initialization pattern
 * - FIXED: Heatmap rendering performance (ImageData instead of fillRect)
 * - FIXED: Error handling for missing data
 * - FIXED: Data validation and sanitization
 */

// ============================================================================
// MODULE 0: CONFIGURATION & STATE
// ============================================================================

const CONFIG = {
  // API
  API_BASE: './api/data',
  WIND_DATA_URL: './api/data/wind_simulation/current.json',
  METADATA_URL: './api/data/system/metadata.json',
  
  // Map
  DEFAULT_LAT: 52.2297,
  DEFAULT_LNG: 21.0122,
  DEFAULT_ZOOM: 13,
  
  // Visualization
  PARTICLE_COUNT: 1200,
  PARTICLE_LIFESPAN: 700,
  PARTICLE_SPEED_SCALE: 0.35,
  STREAMLINE_COUNT: 250,
  HEATMAP_COLORMAP: 'jet',
  
  // Performance
  TARGET_FPS: 30,
  ENABLE_PERFORMANCE_TRACKING: true,
};

let STATE = {
  map: null,
  layers: {
    heatmap: null,
    streamlines: null,
    particles: null,
  },
  data: {
    windSimulation: null,
    lastUpdate: null,
  },
  ui: {
    isDarkMode: localStorage.getItem('darkMode') === 'true',
  },
};

// ============================================================================
// MODULE 1: UTILITIES
// ============================================================================

/**
 * Colormap function – convert [0,1] value to RGB
 */
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
    // Default viridis-like
    r = Math.round(255 * t);
    g = Math.round(255 * (1 - Math.abs(t - 0.5)));
    b = Math.round(255 * (1 - t));
  }
  
  return { r, g, b, a: 0.7 };
}

/**
 * Performance monitoring
 */
class PerformanceMonitor {
  constructor() {
    this.frameCount = 0;
    this.fps = 0;
    this.lastTime = Date.now();
    this.samples = [];
  }
  
  tick() {
    this.frameCount++;
    const now = Date.now();
    
    if (now - this.lastTime >= 500) {
      this.fps = (this.frameCount / ((now - this.lastTime) / 1000)).toFixed(1);
      this.samples.push(parseFloat(this.fps));
      if (this.samples.length > 60) this.samples.shift();
      this.frameCount = 0;
      this.lastTime = now;
      return true;
    }
    return false;
  }
  
  getAvg() {
    if (this.samples.length === 0) return '0';
    const avg = this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
    return avg.toFixed(1);
  }
}

const perfMonitor = new PerformanceMonitor();

// ============================================================================
// MODULE 2: DATA LOADING & VALIDATION
// ============================================================================

/**
 * Load wind simulation data from API – FIXED PATH
 */
async function loadWindData() {
  try {
    console.log(`📥 Loading wind data from: ${CONFIG.WIND_DATA_URL}`);
    
    const response = await fetch(CONFIG.WIND_DATA_URL);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (!validateData(data)) {
      throw new Error('Data validation failed');
    }
    
    STATE.data.windSimulation = data;
    STATE.data.lastUpdate = new Date().toISOString();
    
    console.log('✅ Wind data loaded:', {
      vectors: data.vector_field?.length || 0,
      streamlines: data.streamlines?.length || 0,
      particles: data.particles?.length || 0,
    });
    
    return data;
  } catch (error) {
    console.error('❌ Failed to load wind data:', error);
    return null;
  }
}

/**
 * Validate simulation data
 */
function validateData(data) {
  const errors = [];
  
  if (!data || typeof data !== 'object') errors.push('Data is not an object');
  if (!data.vector_field || !Array.isArray(data.vector_field)) errors.push('Missing/invalid vector_field');
  if (!data.flow_statistics || typeof data.flow_statistics !== 'object') errors.push('Missing/invalid flow_statistics');
  
  if (errors.length > 0) {
    console.error('❌ Data validation errors:', errors);
    return false;
  }
  
  // Ensure magnitude_grid exists
  if (!data.magnitude_grid && data.vector_field.length > 0) {
    console.warn('⚠️  Generating magnitude_grid from vector_field...');
    
    const maxX = Math.max(...data.vector_field.map(v => v.x || v.pixel_x || 0));
    const maxY = Math.max(...data.vector_field.map(v => v.y || v.pixel_y || 0));
    const w = Math.ceil(maxX / 5) + 1;
    const h = Math.ceil(maxY / 5) + 1;
    
    data.magnitude_grid = Array(h).fill().map(() => Array(w).fill(0));
    
    data.vector_field.forEach(v => {
      const i = Math.floor((v.x || v.pixel_x || 0) / 5);
      const j = Math.floor((v.y || v.pixel_y || 0) / 5);
      if (i >= 0 && i < w && j >= 0 && j < h) {
        data.magnitude_grid[j][i] = v.magnitude || 0;
      }
    });
  }
  
  console.log('✅ Data validation passed');
  return true;
}

// ============================================================================
// MODULE 3: VISUALIZATION LAYERS
// ============================================================================

/**
 * Heatmap Layer – FIXED PERFORMANCE
 */
class HeatmapLayer {
  constructor(data, bounds) {
    this.data = data;
    this.bounds = bounds;
    this.canvas = null;
    this.ctx = null;
    this.map = null;
  }
  
  addTo(map) {
    this.map = map;
    
    this.canvas = L.DomUtil.create('canvas', 'heatmap-canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(this.canvas);
    
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    
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
    if (!this.data.magnitude_grid) return;
    
    const grid = this.data.magnitude_grid;
    const h = grid.length;
    const w = grid[0]?.length || 0;
    
    if (h === 0 || w === 0) return;
    
    const stats = this.data.flow_statistics || {};
    const minMag = stats.min_magnitude || 0;
    const maxMag = stats.max_magnitude || 1;
    
    const cellWidth = (this.bounds.getEast() - this.bounds.getWest()) / w;
    const cellHeight = (this.bounds.getNorth() - this.bounds.getSouth()) / h;
    
    const imageData = this.ctx.createImageData(w, h);
    const data = imageData.data;
    
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const value = grid[j]?.[i] ?? 0;
        const normalized = (value - minMag) / (maxMag - minMag || 1);
        const color = getColor(normalized, CONFIG.HEATMAP_COLORMAP);
        
        const idx = (j * w + i) * 4;
        data[idx] = color.r;
        data[idx + 1] = color.g;
        data[idx + 2] = color.b;
        data[idx + 3] = color.a * 255;
      }
    }
    
    this.ctx.putImageData(imageData, 0, 0);
  }
}

/**
 * Streamline Layer
 */
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
    this.ctx.strokeStyle = 'rgba(60, 120, 160, 0.6)';
    this.ctx.lineWidth = 1.8;
    this.ctx.lineCap = 'round';
    
    this.data.streamlines.forEach(streamline => {
      if (streamline.length < 2) return;
      
      this.ctx.beginPath();
      
      streamline.forEach((point, idx) => {
        const lat = point.latitude || point.lat;
        const lng = point.longitude || point.lng;
        
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

/**
 * Particle Animation Layer
 */
class ParticleLayer {
  constructor(data, bounds) {
    this.data = data;
    this.bounds = bounds;
    this.particles = [];
    this.animationId = null;
  }
  
  addTo(map) {
    this.map = map;
    
    this.canvas = L.DomUtil.create('canvas', 'particle-canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(this.canvas);
    
    this.ctx = this.canvas.getContext('2d');
    this.ctx.globalCompositeOperation = 'lighter';
    
    map.on('moveend zoomend resize', () => this._reset(), this);
    this._reset();
    this._initParticles();
    this._animate();
    
    return this;
  }
  
  remove() {
    if (this.animationId) cancelAnimationFrame(this.animationId);
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
  }
  
  _initParticles() {
    this.particles = [];
    
    const vectorField = this.data.vector_field || [];
    if (vectorField.length === 0) return;
    
    for (let i = 0; i < CONFIG.PARTICLE_COUNT; i++) {
      const start = vectorField[Math.floor(Math.random() * vectorField.length)];
      this.particles.push({
        lat: start.latitude || start.lat || 52.2297,
        lng: start.longitude || start.lng || 21.0122,
        vx: start.vx || 0,
        vy: start.vy || 0,
        age: 0,
        maxAge: CONFIG.PARTICLE_LIFESPAN,
      });
    }
  }
  
  _animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.particles.forEach(p => {
      p.age++;
      
      // Update position
      p.lat += p.vy * 0.0001;
      p.lng += p.vx * 0.0001;
      
      // Add diffusion
      p.vx += (Math.random() - 0.5) * 0.01;
      p.vy += (Math.random() - 0.5) * 0.01;
      
      // Draw
      const opacity = 1 - (p.age / p.maxAge);
      this.ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
      
      const point = this.map.latLngToContainerPoint([p.lat, p.lng]);
      this.ctx.fillRect(point.x - 0.5, point.y - 0.5, 1, 1);
      
      // Reset when dead
      if (p.age > p.maxAge) {
        const start = this.data.vector_field[Math.floor(Math.random() * this.data.vector_field.length)];
        p.lat = start.latitude || start.lat || 52.2297;
        p.lng = start.longitude || start.lng || 21.0122;
        p.age = 0;
      }
    });
    
    // FPS monitoring
    if (perfMonitor.tick()) {
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      this.ctx.font = 'bold 11px monospace';
      this.ctx.fillText(`FPS: ${perfMonitor.fps}`, 10, 20);
    }
    
    this.animationId = requestAnimationFrame(() => this._animate());
  }
}

// ============================================================================
// MODULE 4: MAP INITIALIZATION
// ============================================================================

/**
 * Initialize map with all layers
 */
async function initializeMap() {
  console.log('🗺️  Initializing map...');
  
  const mapContainer = document.getElementById('wind-map');
  if (!mapContainer) {
    console.error('❌ Map container not found');
    return;
  }
  
  STATE.map = L.map(mapContainer).setView([CONFIG.DEFAULT_LAT, CONFIG.DEFAULT_LNG], CONFIG.DEFAULT_ZOOM);
  
  // Base layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    opacity: 0.7,
  }).addTo(STATE.map);
  
  // Load data
  const windData = await loadWindData();
  if (!windData) {
    console.error('❌ Cannot load wind data');
    mapContainer.innerHTML = '<p style="padding: 20px; color: red;">Failed to load simulation data. Check console.</p>';
    return;
  }
  
  // Calculate bounds
  const lats = windData.vector_field.map(v => v.latitude || v.lat).filter(Boolean);
  const lngs = windData.vector_field.map(v => v.longitude || v.lng).filter(Boolean);
  
  const bounds = L.latLngBounds(
    [Math.min(...lats), Math.min(...lngs)],
    [Math.max(...lats), Math.max(...lngs)]
  );
  
  // Add layers
  STATE.layers.heatmap = new HeatmapLayer(windData, bounds).addTo(STATE.map);
  STATE.layers.streamlines = new StreamlineLayer(windData, bounds).addTo(STATE.map);
  STATE.layers.particles = new ParticleLayer(windData, bounds).addTo(STATE.map);
  
  STATE.map.fitBounds(bounds);
  
  console.log('✅ Map initialized');
  updateMetricsPanel(windData);
}

/**
 * Update metrics display
 */
function updateMetricsPanel(data) {
  const stats = data.flow_statistics || {};
  const perf = data.performance || {};
  
  const panel = document.getElementById('metrics-panel');
  if (!panel) return;
  
  panel.innerHTML = `
    <div class="metrics-grid">
      <div class="metric-card">
        <span class="metric-label">Max Speed</span>
        <span class="metric-value">${(stats.max_magnitude || 0).toFixed(2)} m/s</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Mean Speed</span>
        <span class="metric-value">${(stats.mean_magnitude || 0).toFixed(2)} m/s</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Turbulence</span>
        <span class="metric-value">${(stats.turbulence_intensity || 0).toFixed(3)}</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Compute Time</span>
        <span class="metric-value">${(perf.simulation_time || 0).toFixed(2)}s</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Grid Cells/s</span>
        <span class="metric-value">${(perf.grid_cells_per_second || 0).toLocaleString()}</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">FPS</span>
        <span class="metric-value">${perfMonitor.getAvg()}</span>
      </div>
    </div>
  `;
}

// ============================================================================
// MODULE 5: INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('✅ Page loaded, initializing...');
  
  initializeMap().catch(err => {
    console.error('❌ Initialization failed:', err);
    document.getElementById('wind-map').innerHTML = `
      <div style="padding: 20px; color: red; font-family: monospace;">
        <strong>Error:</strong> ${err.message}
        <br><br>
        <strong>Troubleshooting:</strong>
        <ul>
          <li>Check that ./api/data/wind_simulation/current.json exists</li>
          <li>Check browser console (F12) for CORS errors</li>
          <li>Ensure data was generated and pushed to GitHub</li>
        </ul>
      </div>
    `;
  });
});

// Export for external use
window.STATE = STATE;
window.loadWindData = loadWindData;
window.initializeMap = initializeMap;
