# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "weather")
RAW_COTTON_PATH = os.path.join(BASE_DIR, "data", "raw", "cotton_dataset.xls")
DB_PATH         = os.path.join(BASE_DIR, "data", "cotton_project.duckdb")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
FIGURES_DIR     = os.path.join(BASE_DIR, "reports", "figures")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# ── Weather API ───────────────────────────────────────────────────────────────
WEATHER_START_YEAR = 2000
WEATHER_END_YEAR   = 2026
TIMEZONE           = "Asia/Baku"

WEATHER_VARIABLES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max"
]  

# ── Locations ───────────────────────────────────────────────────────────────── 

LOCATIONS = { 
    "Goranboy district": {"lat": 40.60446822949957, "lon": 46.78557757111397}, 
    "Kurdamir district": {"lat": 40.35398237909627, "lon":  48.17170737698027},   
    "Yevlakh district": {"lat": 40.62045643709087, "lon": 47.14562733367288},  
    "Zardab district": {"lat": 40.217812657916, "lon": 47.7093089349384}, 
    "Tartar district": {"lat": 40.34241192193064, "lon": 46.932557394348656},  
    "Aghdam district": {"lat": 39.992604060390846, "lon": 46.933141466775474},  
    "Sabirabad district": {"lat": 39.98712358421815, "lon": 48.46948476123727},  
    "Saatli district": {"lat": 39.90942890881128, "lon": 48.35726326040041}, 
    "Imishli district": {"lat": 39.869646771846185, "lon": 48.066732555170375},  
    "Beylagan district": {"lat": 39.77182056637182, "lon": 47.618528720693746},  
    "Aghjabadi district": {"lat": 40.04919485497778, "lon": 47.458862897032425},  
    "Barda district": {"lat": 40.37072912330946, "lon": 47.137354717029744},  
    "Bilasuvar district": {"lat": 39.456963024994344, "lon": 48.547163785792866},  
    "Neftchala district": {"lat": 39.388728158677544, "lon": 49.24136530498952}, 
    "Salyan district" : {"lat": 39.59395458602429, "lon": 48.98129523365497}  
}

# ── District → Weather Station mapping ───────────────────────────────────────
REGION_TO_WEATHER = { 
    "Goranboy district",  
    "Kurdamir district",  
    "Yevlakh district",  
    "Zardab district",  
    "Tartar district",  
    "Aghdam district",  
    "Sabirabad district",  
    "Saatli district",  
    "Imishli district",  
    "Beylagan district",  
    "Aghjabadi district",  
    "Barda district",  
    "Bilasuvar district",  
    "Neftchala district", 
    "Salyan district" 
}

# ── Agronomic constants ───────────────────────────────────────────────────────
COTTON_BASE_TEMP = 15.5

STAGES = {
    "planting":     (3, 4),
    "growing":      (5, 8),
    "boll_forming": (8, 9),
    "harvest":      (9, 11),
}

# ── Model settings ────────────────────────────────────────────────────────────
TRAIN_UNTIL_YEAR  = 2021
PREDICT_YEARS     = [2025] 