import os; import requests; import subprocess; from datetime import datetime; import argparse; from modules import skrypt0_lidar, skrypt1_podtopienia, skrypt2_wiatr, skrypt3_mwc
CONFIG = {"location": {"latitude": 54.16, "longitude": 19.40, "timezone": "Europe/Warsaw"}, "paths": {"nmt": "dane/nmt.tif", "nmpt": "dane/nmpt.tif", "landcover": "dane/landcover.tif", "bdot_zip": "dane/bdot10k.zip", "laz_folder": "dane/laz/", "output_folder": "wyniki/", "bdot_extract": "wyniki/bdot_extracted", "output_crowns_raster": "wyniki/rastry/korony_drzew.tif", "output_trees_vector": "wyniki/wektory/inwentaryzacja_drzew.gpkg", "output_flood_raster": "wyniki/rastry/max_glebokosc_wody.tif", "output_wind_raster": "wyniki/rastry/predkosc_wiatru.tif", "output_utci_raster": "wyniki/rastry/komfort_cieplny_utci.tif", "output_flood_tiles": "wyniki/kafelki/podtopienia", "output_wind_tiles": "wyniki/kafelki/wiatr", "output_utci_tiles": "wyniki/kafelki/komfort"}, "params": {"lidar": {"target_res": 1.0, "min_tree_height": 4.0, "treetop_filter_size": 7, "min_crown_area_m2": 10, "max_plausible_tree_height": 100.0, "crown_base_factor": 0.25}, "flood": {"target_res": 5.0, "dt_s": 10.0, "total_rainfall_mm": 70.0, "interception_mm": 5.0, "rainfall_duration_h": 2.0, "simulation_duration_h": 4.0, "obstacle_height_m": 2.5, "manning_map": {1: 0.015, 2: 1000, 3: 0.100, 5: 0.035, 6: 0.025, 7: 0.030, 'default': 0.05}, "cn_map": {1: 98, 2: 98, 3: 55, 5: 70, 6: 77, 7: 100, 'default': 75}}, "wind": {"target_res": 3.0, "analysis_height": 2.0, "building_threshold": 2.0, "bdot_building_file": "BUBD_A.gpkg", "z0_map": { 1: 0.05, 2: 0.8, 3: 1.0, 5: 0.03, 6: 0.01, 7: 0.0002, -1: 0.1 }}, "uhi": {"simulation_datetime": datetime.now(), "solar_constant": 1361, "atmospheric_transmissivity": 0.75, "lst_base_temp": { 1: 42, 2: 38, 3: 32, 5: 34, 6: 40, 7: 28, -1: 36 }, "insolation_heating_factor": 0.01, "mrt_insolation_factor": 0.028}}}
def get_live_weather(lat, lon, timezone):
    print("-> Pobieranie aktualnych danych pogodowych...")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&timezone={timezone}"
    try:
        data = requests.get(url, timeout=10).json()['current']
        print(f"  -> Temp: {data['temperature_2m']}°C, Wilg.: {data['relative_humidity_2m']}%, Wiatr: {data['wind_speed_10m']:.1f} km/h, Kier.: {data['wind_direction_10m']}°")
        return {"temperature": data['temperature_2m'], "humidity": data['relative_humidity_2m'], "wind_speed": data['wind_speed_10m']/3.6, "wind_direction": data['wind_direction_10m']}
    except Exception as e:
        print(f"BŁĄD: Nie można pobrać danych pogodowych: {e}. Używam wartości domyślnych.")
        return {"temperature": 25, "humidity": 50, "wind_speed": 5.0, "wind_direction": 270}
def generate_tiles(input_raster, output_folder, zoom_levels='10-16'):
    if not os.path.exists(input_raster): print(f"BŁĄD: Plik wejściowy dla kafelków nie istnieje: {input_raster}"); return
    print(f"-> Generowanie kafelków dla: {os.path.basename(input_raster)}")
    if os.path.exists(output_folder): import shutil; shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    command = ['gdal2tiles.py', '--profile', 'mercator', '-z', zoom_levels, '-w', 'none', '--s_srs', str(rasterio.open(input_raster).crs), input_raster, output_folder]
    subprocess.run(command)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Uruchamia wybrane moduły analizy GIS.")
    parser.add_argument('analiza', choices=['lidar', 'podtopienia', 'wiatr', 'mwc', 'wszystko'], help="Wybierz analizę do uruchomienia.")
    args = parser.parse_args()
    for folder in ["wyniki/rastry", "wyniki/kafelki", "wyniki/wektory"]: os.makedirs(folder, exist_ok=True)
    if args.analiza in ['lidar', 'wszystko']: skrypt0_lidar.run_lidar_processing(CONFIG)
    if args.analiza in ['podtopienia', 'wszystko']:
        result_raster = skrypt1_podtopienia.run_flood_analysis(CONFIG); generate_tiles(result_raster, CONFIG['paths']['output_flood_tiles'])
    if args.analiza in ['wiatr', 'wszystko']:
        weather = get_live_weather(CONFIG['location']['latitude'], CONFIG['location']['longitude'], CONFIG['location']['timezone'])
        CONFIG['params']['wind']['wind_speed'] = weather['wind_speed']; CONFIG['params']['wind']['wind_direction'] = weather['wind_direction']
        result_raster = skrypt2_wiatr.run_wind_analysis(CONFIG); generate_tiles(result_raster, CONFIG['paths']['output_wind_tiles'])
    if args.analiza in ['mwc', 'wszystko']:
        if not os.path.exists(CONFIG['paths']['output_wind_raster']):
            print("-> OSTRZEŻENIE: Brak wyników z analizy wiatru. Uruchamiam ją najpierw...")
            weather = get_live_weather(CONFIG['location']['latitude'], CONFIG['location']['longitude'], CONFIG['location']['timezone'])
            CONFIG['params']['wind']['wind_speed'] = weather['wind_speed']; CONFIG['params']['wind']['wind_direction'] = weather['wind_direction']
            skrypt2_wiatr.run_wind_analysis(CONFIG)
        weather = get_live_weather(CONFIG['location']['latitude'], CONFIG['location']['longitude'], CONFIG['location']['timezone'])
        result_raster = skrypt3_mwc.run_uhi_analysis(CONFIG, weather); generate_tiles(result_raster, CONFIG['paths']['output_utci_tiles'])
    print("\n✅ Wybrane zadania zakończone!")