# -*- coding: utf-8 -*-
"""
Complete Wind Simulation Script - Standalone Version
Version: 2.3.2 - FIXED coordinate transformation for OSM compatibility
"""

import os, sys, time, json, shutil, subprocess, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from numba import njit, prange
from typing import Dict, List, Tuple, Optional
from pyproj import Transformer

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
DRIVE_BASE = "/content/drive/MyDrive/ProjektGIS/dane"
PATHS = {
    "nmt"      : f"{DRIVE_BASE}/nmt.tif",
    "nmpt"     : f"{DRIVE_BASE}/nmpt.tif",
    "buildings": f"{DRIVE_BASE}/buildings.gpkg",
}

SIM_PARAMS = {
    "grid_width"       : 750,
    "buffer_size"      : 50,
    "max_iterations"   : 4000,
    "relaxation_rate"  : 1.4,
    "height_threshold" : 2.5,
    "vector_stride"    : 5,
    "output_precision" : 4,
    "generate_streamlines": True,
    "generate_particles": True,
    "streamline_count": 500,
    "particle_count": 100
}

OUTPUT_DIR = Path("/content/wind_simulation_output")

# ──────────────────────────────────────────────────────────────────────────
# LBM SIMULATION MODULE (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True)
def _lbm_enhanced(mask, wind_speed, wind_deg, nx, ny, max_iter, omega,
                  enable_performance_tracking=False):
    """Enhanced LBM kernel - NUMBA COMPATIBLE"""
    
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
    
    cx, cy = c[:, 0], c[:, 1]
    
    F = np.ones((ny, nx, 9), dtype=np.float64)
    Fs = np.empty((ny, nx, 9), dtype=np.float64)
    
    rad = np.deg2rad(90.0 - wind_deg)
    u0 = 0.1 * np.cos(rad)
    v0 = 0.1 * np.sin(rad)
    
    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    
    convergence_history = np.zeros(max_iter, dtype=np.float64) if enable_performance_tracking else None
    
    for iteration in range(max_iter):
        
        for j in prange(ny):
            for i in range(nx):
                for k in range(9):
                    src_j = (j - cy[k]) % ny
                    src_i = (i - cx[k]) % nx
                    Fs[j, i, k] = F[src_j, src_i, k]
        
        F, Fs = Fs, F
        
        for j in prange(ny):
            for i in range(nx):
                if mask[j, i]:
                    temp = F[j, i, 1]; F[j, i, 1] = F[j, i, 3]; F[j, i, 3] = temp
                    temp = F[j, i, 2]; F[j, i, 2] = F[j, i, 4]; F[j, i, 4] = temp
                    temp = F[j, i, 5]; F[j, i, 5] = F[j, i, 7]; F[j, i, 7] = temp
                    temp = F[j, i, 6]; F[j, i, 6] = F[j, i, 8]; F[j, i, 8] = temp
        
        for i in prange(nx):
            for k in range(9):
                F[0, i, k] = F[1, i, k]
                F[ny-1, i, k] = F[ny-2, i, k]
        
        for j in prange(ny):
            for k in range(9):
                F[j, 0, k] = F[j, 1, k]
                F[j, nx-1, k] = F[j, nx-2, k]
        
        for j in prange(ny):
            for i in range(nx):
                s_rho = s_ux = s_uy = 0.0
                
                for k in range(9):
                    f_val = F[j, i, k]
                    s_rho += f_val
                    s_ux += f_val * cx[k]
                    s_uy += f_val * cy[k]
                
                rho[j, i] = s_rho
                if s_rho > 1e-12:
                    ux[j, i] = s_ux / s_rho
                    uy[j, i] = s_uy / s_rho
                else:
                    ux[j, i] = uy[j, i] = 0.0
        
        if wind_deg >= 315 or wind_deg < 45:
            for i in prange(nx):
                ux[0, i] = u0
                uy[0, i] = v0
        elif wind_deg < 135:
            for j in prange(ny):
                ux[j, nx-1] = u0
                uy[j, nx-1] = v0
        elif wind_deg < 225:
            for i in prange(nx):
                ux[ny-1, i] = u0
                uy[ny-1, i] = v0
        else:
            for j in prange(ny):
                ux[j, 0] = u0
                uy[j, 0] = v0
        
        for j in prange(ny):
            for i in range(nx):
                u_local = ux[j, i]
                v_local = uy[j, i]
                rho_local = rho[j, i]
                usq = u_local * u_local + v_local * v_local
                
                for k in range(9):
                    cu = u_local * cx[k] + v_local * cy[k]
                    feq = rho_local * w[k] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
                    F[j, i, k] += omega * (feq - F[j, i, k])
        
        if enable_performance_tracking and iteration % 10 == 0:
            velocity_magnitude = 0.0
            for j in range(ny):
                for i in range(nx):
                    velocity_magnitude += ux[j, i]*ux[j, i] + uy[j, i]*uy[j, i]
            convergence_history[iteration] = velocity_magnitude / (nx * ny)
    
    scale = wind_speed / 0.1
    ux *= scale
    uy *= scale
    
    return ux, uy, convergence_history

@njit(fastmath=True)
def _generate_streamlines(ux, uy, nx, ny, num_streamlines=200, max_points=100,
                         min_speed=0.1, step_size=1.0):
    """Generate streamlines using integration"""
    streamlines = []
    
    for stream_idx in range(num_streamlines):
        start_x = np.random.random() * (nx - 1)
        start_y = np.random.random() * (ny - 1)
        
        streamline = []
        x, y = start_x, start_y
        
        for point_idx in range(max_points):
            if x < 1 or x >= nx-1 or y < 1 or y >= ny-1:
                break
            
            i, j = int(x), int(y)
            fx, fy = x - i, y - j
            
            u00 = ux[j, i]
            u10 = ux[j, i+1]
            u01 = ux[j+1, i]
            u11 = ux[j+1, i+1]
            
            v00 = uy[j, i]
            v10 = uy[j, i+1]
            v01 = uy[j+1, i]
            v11 = uy[j+1, i+1]
            
            u_interp = u00*(1-fx)*(1-fy) + u10*fx*(1-fy) + u01*(1-fx)*fy + u11*fx*fy
            v_interp = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy
            
            speed = np.sqrt(u_interp*u_interp + v_interp*v_interp)
            
            if speed < min_speed:
                break
            
            streamline.append((x, y, speed))
            
            k1x, k1y = u_interp * step_size, v_interp * step_size
            x += k1x
            y += k1y
        
        if len(streamline) > 5:
            streamlines.append(streamline)
    
    return streamlines

@njit(fastmath=True)
def _generate_particle_paths(ux, uy, nx, ny, num_particles=1000, max_steps=200,
                            dt=0.1, min_speed=0.05):
    """Generate particle trajectories"""
    particles = []
    
    for particle_idx in range(num_particles):
        start_x = np.random.random() * (nx - 1)
        start_y = np.random.random() * (ny - 1)
        
        path = []
        x, y = start_x, start_y
        age = 0
        
        for step in range(max_steps):
            if x < 1 or x >= nx-1 or y < 1 or y >= ny-1:
                break
            
            i, j = int(x), int(y)
            fx, fy = x - i, y - j
            
            u00 = ux[j, i]; u10 = ux[j, i+1]; u01 = ux[j+1, i]; u11 = ux[j+1, i+1]
            v00 = uy[j, i]; v10 = uy[j, i+1]; v01 = uy[j+1, i]; v11 = uy[j+1, i+1]
            
            u_interp = u00*(1-fx)*(1-fy) + u10*fx*(1-fy) + u01*(1-fx)*fy + u11*fx*fy
            v_interp = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy
            
            speed = np.sqrt(u_interp*u_interp + v_interp*v_interp)
            
            if speed < min_speed:
                break
            
            diffusion = 0.1
            u_interp += np.random.normal(0, diffusion)
            v_interp += np.random.normal(0, diffusion)
            
            path.append((x, y, u_interp, v_interp, speed, age))
            
            x += u_interp * dt
            y += v_interp * dt
            age += 1
        
        if len(path) > 3:
            particles.append(path)
    
    return particles

def run_wind_simulation(obstacle_mask: np.ndarray,
                       grid_info: Dict,
                       weather_data: Dict,
                       sim_params: Dict) -> Dict:
    """Enhanced wind simulation"""
    
    print(f"🌬️  Starting enhanced wind simulation...")
    print(f"    Grid size: {obstacle_mask.shape}")
    print(f"    Wind: {weather_data['wind_speed_ms']} m/s @ {weather_data['wind_direction_deg']}°")
    
    ny, nx = obstacle_mask.shape
    t_start = time.time()
    
    enable_performance_tracking = sim_params.get("enable_performance_tracking", False)
    generate_streamlines = sim_params.get("generate_streamlines", True)
    generate_particles = sim_params.get("generate_particles", True)
    
    ux, uy, convergence_history = _lbm_enhanced(
        obstacle_mask,
        weather_data["wind_speed_ms"],
        weather_data["wind_direction_deg"],
        nx, ny,
        sim_params["max_iterations"],
        sim_params["relaxation_rate"],
        enable_performance_tracking
    )
    
    simulation_time = time.time() - t_start
    print(f"✅ LBM simulation completed in {simulation_time:.2f}s")
    
    magnitude = np.sqrt(ux**2 + uy**2)
    
    buffer_size = sim_params.get("buffer_size", 0)
    if buffer_size > 0:
        core_mag = magnitude[buffer_size:-buffer_size, buffer_size:-buffer_size]
        core_ux = ux[buffer_size:-buffer_size, buffer_size:-buffer_size]
        core_uy = uy[buffer_size:-buffer_size, buffer_size:-buffer_size]
    else:
        core_mag = magnitude
        core_ux = ux
        core_uy = uy
    
    stats = {
        "min_magnitude": float(np.min(core_mag)),
        "max_magnitude": float(np.max(core_mag)),
        "mean_magnitude": float(np.mean(core_mag)),
        "std_magnitude": float(np.std(core_mag)),
        "median_magnitude": float(np.median(core_mag)),
        "percentile_95": float(np.percentile(core_mag, 95)),
        "percentile_05": float(np.percentile(core_mag, 5)),
        "percentile_75": float(np.percentile(core_mag, 75)),
        "percentile_25": float(np.percentile(core_mag, 25)),
        "mean_vorticity": float(np.mean(np.abs(
            np.gradient(core_uy, axis=1) - np.gradient(core_ux, axis=0)
        ))),
        "turbulence_intensity": float(np.std(core_mag) / np.mean(core_mag)) if np.mean(core_mag) > 0 else 0.0
    }
    
    print(f"📊 Flow statistics calculated")
    
    stride = sim_params.get("vector_stride", 5)
    precision = sim_params.get("output_precision", 4)
    
    vectors = []
    for y in range(0, ny, stride):
        for x in range(0, nx, stride):
            if not obstacle_mask[y, x]:
                vectors.append({
                    "x": int(x),
                    "y": int(y),
                    "vx": round(float(ux[y, x]), precision),
                    "vy": round(float(uy[y, x]), precision),
                    "magnitude": round(float(magnitude[y, x]), precision)
                })
    
    print(f"🎯 Generated {len(vectors)} vector field points")
    
    streamlines_data = []
    if generate_streamlines:
        t_streamlines = time.time()
        streamlines = _generate_streamlines(
            ux, uy, nx, ny,
            num_streamlines=sim_params.get("streamline_count", 200),
            max_points=sim_params.get("streamline_max_points", 100)
        )
        
        for streamline in streamlines:
            streamline_points = []
            for point in streamline:
                streamline_points.append({
                    "x": round(point[0], precision),
                    "y": round(point[1], precision),
                    "speed": round(point[2], precision)
                })
            streamlines_data.append(streamline_points)
        
        streamlines_time = time.time() - t_streamlines
        print(f"🌊 Generated {len(streamlines_data)} streamlines in {streamlines_time:.2f}s")
    
    particles_data = []
    if generate_particles:
        t_particles = time.time()
        particles = _generate_particle_paths(
            ux, uy, nx, ny,
            num_particles=sim_params.get("particle_count", 1000),
            max_steps=sim_params.get("particle_max_steps", 200)
        )
        
        for particle_path in particles:
            path_points = []
            for point in particle_path:
                path_points.append({
                    "x": round(point[0], precision),
                    "y": round(point[1], precision),
                    "vx": round(point[2], precision),
                    "vy": round(point[3], precision),
                    "speed": round(point[4], precision),
                    "age": int(point[5])
                })
            particles_data.append(path_points)
        
        particles_time = time.time() - t_particles
        print(f"🔴 Generated {len(particles_data)} particle paths in {particles_time:.2f}s")
    
    total_time = time.time() - t_start
    
    results = {
        "metadata": {
            "version": "2.3.2",
            "enhanced_features": True,
            "timestamp": time.time(),
            "computation_time": round(total_time, 2),
            "simulation_time": round(simulation_time, 2)
        },
        "performance": {
            "total_time": round(total_time, 2),
            "simulation_time": round(simulation_time, 2),
            "post_processing_time": round(total_time - simulation_time, 2),
            "iterations_per_second": round(sim_params["max_iterations"] / simulation_time, 1),
            "grid_cells_per_second": round((nx * ny * sim_params["max_iterations"]) / simulation_time, 0)
        },
        "flow_statistics": stats,
        "vector_field": vectors,
        "magnitude_grid": magnitude.round(precision).tolist(),
        "streamlines": streamlines_data if generate_streamlines else [],
        "particles": particles_data if generate_particles else [],
        "convergence_history": convergence_history.tolist() if convergence_history is not None else []
    }
    
    print(f"🎉 Enhanced simulation completed successfully!")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Performance: {results['performance']['iterations_per_second']:.1f} iter/s")
    print(f"    Grid cells/s: {results['performance']['grid_cells_per_second']:,.0f}")
    
    return results

# ──────────────────────────────────────────────────────────────────────────
# DATA PREPARATION (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────

def mount_drive():
    """Montuje Google Drive"""
    from google.colab import drive
    print("💾 Mounting Google Drive…")
    drive.mount('/content/drive')
    print("✅ Drive mounted")

def validate_inputs():
    """Waliduje dostępność plików wejściowych"""
    missing = [k for k,p in PATHS.items() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    for k,p in PATHS.items():
        sz = os.path.getsize(p)/(1024*1024)
        print(f"✅ {k}: {sz:.1f} MB")

def align_rasters():
    """Wyrównuje rastry NMT i NMPT do wspólnego układu"""
    TMP_DIR = Path("/content/temp_sim")
    TMP_DIR.mkdir(exist_ok=True)
    
    with rasterio.open(PATHS["nmt"]) as dtm, rasterio.open(PATHS["nmpt"]) as dsm:
        dtm_arr, dsm_arr = dtm.read(1), dsm.read(1)
        
        if dtm_arr.shape != dsm_arr.shape:
            h,w = min(dtm_arr.shape[0],dsm_arr.shape[0]), min(dtm_arr.shape[1],dsm_arr.shape[1])
            dtm_arr, dsm_arr = dtm_arr[:h,:w], dsm_arr[:h,:w]
            tr = from_bounds(dtm.bounds.left, dtm.bounds.top-h*abs(dtm.transform[4]),
                             dtm.bounds.left+w*dtm.transform[0], dtm.bounds.top,
                             w, h)
            prof = dtm.profile
            prof.update(height=h, width=w, transform=tr)
        else:
            prof, h, w = dtm.profile, *dtm_arr.shape
        
        nmt_p = TMP_DIR/"nmt.tif"
        nmpt_p = TMP_DIR/"nmpt.tif"
        with rasterio.open(nmt_p,'w',**prof) as dst:
            dst.write(dtm_arr,1)
        with rasterio.open(nmpt_p,'w',**prof) as dst:
            dst.write(dsm_arr,1)
    
    shutil.copy2(PATHS["buildings"], TMP_DIR/"buildings.gpkg")
    
    return {
        "nmt": nmt_p,
        "nmpt": nmpt_p,
        "buildings": TMP_DIR/"buildings.gpkg",
        "profile": prof,
        "h": h,
        "w": w,
        "dtm": dtm_arr,
        "dsm": dsm_arr
    }

def make_obstacle_mask(aligned_data):
    """Tworzy maskę przeszkód"""
    dtm = aligned_data["dtm"]
    dsm = aligned_data["dsm"]
    prof = aligned_data["profile"]
    
    gdf = gpd.read_file(aligned_data["buildings"])
    if gdf.crs != prof["crs"]:
        gdf = gdf.to_crs(prof["crs"])
    chm = dsm - dtm
    footprint = rasterize(
        gdf.geometry, out_shape=chm.shape,
        transform=prof["transform"],
        fill=0, default_value=1
    )
    mask = (footprint.astype(bool)) & (chm >= SIM_PARAMS["height_threshold"])
    
    buf = SIM_PARAMS["buffer_size"]
    mask_buf = np.pad(mask, buf, constant_values=False)
    
    H2, W2 = mask_buf.shape
    sw = SIM_PARAMS["grid_width"]
    sh = int(sw * H2 / W2)
    
    from scipy.ndimage import zoom
    mask_small = zoom(mask_buf.astype(float), (sh/H2, sw/W2), order=1) > 0.5
    return mask_small

# ──────────────────────────────────────────────────────────────────────────
# COORDINATE TRANSFORMATION (FIXED)
# ──────────────────────────────────────────────────────────────────────────

def create_coordinate_transformer(aligned_data, sim_params):
    """
    FIXED: Coordinate transformation pixel → WGS84
    Compatible with OSM building coordinates
    """
    prof = aligned_data["profile"]
    H, W = aligned_data["h"], aligned_data["w"]
    buf = sim_params["buffer_size"]
    sw = sim_params["grid_width"]
    
    H_buf = H + 2 * buf
    W_buf = W + 2 * buf
    sh = int(sw * H_buf / W_buf)
    
    to_wgs = Transformer.from_crs(prof["crs"], "EPSG:4326", always_xy=True)
    
    raster_transform = prof["transform"]
    
    print(f"🔧 Transform params:")
    print(f"   Grid: {sw}×{sh}, Buffer: {buf}, Original: {W}×{H}")
    
    def pixel_to_geo(sim_x, sim_y):
        """
        sim_x, sim_y: coordinates in simulation grid [0..sw, 0..sh]
        return: (lon, lat) in WGS84
        """
        # 1. Scale to full resolution with buffer
        x_buf = sim_x * (W_buf / sw)
        y_buf = sim_y * (H_buf / sh)
        
        # 2. Remove buffer offset → back to original grid
        x_orig = x_buf - buf
        y_orig = y_buf - buf
        
        # 3. Convert to metric coordinates using ORIGINAL transform
        x_m = raster_transform[2] + x_orig * raster_transform[0] + y_orig * raster_transform[1]
        y_m = raster_transform[5] + x_orig * raster_transform[3] + y_orig * raster_transform[4]
        
        # 4. CRS → WGS84
        lon, lat = to_wgs.transform(x_m, y_m)
        
        return lon, lat
    
    return pixel_to_geo

def save_georeferenced_results(results, aligned_data, sim_params, filename="wind_simulation_results.json"):
    """Save results with WGS84 coordinates (OSM compatible)"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    pixel_to_geo = create_coordinate_transformer(aligned_data, sim_params)
    
    original_crs = str(aligned_data["profile"]["crs"])
    
    H, W = aligned_data["h"], aligned_data["w"]
    buf = sim_params["buffer_size"]
    sw = sim_params["grid_width"]
    H_buf = H + 2 * buf
    W_buf = W + 2 * buf
    sh = int(sw * H_buf / W_buf)
    
    print(f"\n🌍 Calculating geographic bounds...")
    
    corners = [
        pixel_to_geo(0, 0),
        pixel_to_geo(sw-1, 0),
        pixel_to_geo(sw-1, sh-1),
        pixel_to_geo(0, sh-1)
    ]
    
    mid_points = [
        pixel_to_geo(sw//2, 0),
        pixel_to_geo(sw-1, sh//2),
        pixel_to_geo(sw//2, sh-1),
        pixel_to_geo(0, sh//2)
    ]
    
    all_points = corners + mid_points
    lngs = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    
    wgs84_bounds = {
        "west": min(lngs),
        "east": max(lngs),
        "south": min(lats), 
        "north": max(lats)
    }
    
    center_lon, center_lat = pixel_to_geo(sw//2, sh//2)
    
    print(f"   • West:  {wgs84_bounds['west']:.6f}°")
    print(f"   • East:  {wgs84_bounds['east']:.6f}°")
    print(f"   • South: {wgs84_bounds['south']:.6f}°")
    print(f"   • North: {wgs84_bounds['north']:.6f}°")
    print(f"   • Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
    
    if (14.0 <= wgs84_bounds['west'] <= 25.0 and 
        14.0 <= wgs84_bounds['east'] <= 25.0 and
        49.0 <= wgs84_bounds['south'] <= 55.0 and 
        49.0 <= wgs84_bounds['north'] <= 55.0):
        print("   ✅ Coordinates validated: within Poland bounds")
    else:
        print("   ⚠️  WARNING: Coordinates outside expected Poland bounds!")
    
    print(f"\n🔄 Transforming {len(results.get('vector_field', []))} vector field points...")
    if "vector_field" in results:
        for i, point in enumerate(results["vector_field"]):
            if "x" in point and "y" in point:
                try:
                    lng, lat = pixel_to_geo(point["x"], point["y"])
                    point["longitude"] = round(lng, 8)
                    point["latitude"] = round(lat, 8)
                    point["pixel_x"] = int(point["x"])
                    point["pixel_y"] = int(point["y"])
                except Exception as e:
                    print(f"   ⚠️  Error transforming point {i}: {e}")
    
    if "streamlines" in results and len(results["streamlines"]) > 0:
        print(f"🔄 Transforming {len(results['streamlines'])} streamlines...")
        for streamline in results["streamlines"]:
            for point in streamline:
                if "x" in point and "y" in point:
                    try:
                        lng, lat = pixel_to_geo(point["x"], point["y"])
                        point["longitude"] = round(lng, 8)
                        point["latitude"] = round(lat, 8)
                        point["pixel_x"] = int(point["x"])
                        point["pixel_y"] = int(point["y"])
                    except:
                        pass
    
    if "particles" in results and len(results["particles"]) > 0:
        print(f"🔄 Transforming {len(results['particles'])} particle trajectories...")
        for particle_path in results["particles"]:
            for point in particle_path:
                if "x" in point and "y" in point:
                    try:
                        lng, lat = pixel_to_geo(point["x"], point["y"])
                        point["longitude"] = round(lng, 8)
                        point["latitude"] = round(lat, 8)
                        point["pixel_x"] = int(point["x"])
                        point["pixel_y"] = int(point["y"])
                    except:
                        pass
    
    results["spatial_reference"] = {
        "crs": "EPSG:4326",
        "epsg_code": 4326,
        "bounds_wgs84": wgs84_bounds,
        "center_wgs84": {
            "longitude": round(center_lon, 8),
            "latitude": round(center_lat, 8)
        },
        "transformation_info": {
            "source_crs": original_crs,
            "target_crs": "EPSG:4326 (WGS84)",
            "grid_dimensions": {
                "simulation_width": sw,
                "simulation_height": sh,
                "original_width": W,
                "original_height": H,
                "buffer_size": buf
            },
            "scale_factors": {
                "x_scale": round(W_buf / sw, 6),
                "y_scale": round(H_buf / sh, 6)
            },
            "note": "Coordinates compatible with OpenStreetMap and web mapping services"
        }
    }
    
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    metadata_path = OUTPUT_DIR / "spatial_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "bounds": wgs84_bounds,
            "center": results["spatial_reference"]["center_wgs84"],
            "crs": "EPSG:4326",
            "compatible_with": ["OpenStreetMap", "Leaflet", "Google Maps", "Mapbox"]
        }, f, indent=2)
    
    print(f"\n✅ Results with WGS84 coordinates saved to: {output_path}")
    print(f"📋 Spatial metadata saved to: {metadata_path}")
    
    return output_path

# ──────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────

def run_complete_wind_simulation():
    """Execute complete wind simulation pipeline"""
    
    print("🌬️ Starting Complete Wind Simulation Pipeline...")
    total_start = time.time()
    
    try:
        validate_inputs()
        
        print("\n📐 Preparing raster data...")
        aligned_data = align_rasters()
        
        print("\n🏗️ Creating obstacle mask...")
        obstacle_mask = make_obstacle_mask(aligned_data)
        print(f"✅ Obstacle mask created: {obstacle_mask.shape}")
        
        grid_info = {
            "width": obstacle_mask.shape[1],
            "height": obstacle_mask.shape[0]
        }
        
        weather_data = {
            "wind_speed_ms": 3.5,
            "wind_direction_deg": 200,
            "temperature": 20.0,
            "humidity": 60.0
        }
        
        print(f"\n🌤️ Weather conditions:")
        print(f"   Wind: {weather_data['wind_speed_ms']} m/s @ {weather_data['wind_direction_deg']}°")
        
        print("\n🌊 Running wind simulation...")
        results = run_wind_simulation(
            obstacle_mask=obstacle_mask,
            grid_info=grid_info,
            weather_data=weather_data,
            sim_params=SIM_PARAMS
        )
        
        print("\n💾 Saving georeferenced results...")
        output_file = save_georeferenced_results(results, aligned_data, SIM_PARAMS)
        
        total_time = time.time() - total_start
        print(f"\n🎉 Complete simulation finished successfully!")
        print(f"   Total pipeline time: {total_time:.2f}s")
        print(f"   Output file: {output_file}")
        print(f"   Vector field points: {len(results.get('vector_field', []))}")
        print(f"   Streamlines: {len(results.get('streamlines', []))}")
        print(f"   Particle trajectories: {len(results.get('particles', []))}")
        
        return results, output_file
    
    except Exception as e:
        print(f"\n❌ Error in simulation pipeline: {str(e)}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        raise

# ──────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, output_file = run_complete_wind_simulation()
