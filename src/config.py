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
WEATHER_END_YEAR   = 2024
TIMEZONE           = "Asia/Baku"

WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration"
]

# ── Locations ─────────────────────────────────────────────────────────────────
LOCATIONS = {
    "Ganja":      {"lat": 40.6828, "lon": 46.3606},
    "Shamkir":    {"lat": 40.8290, "lon": 46.0170},
    "Sabirabad":  {"lat": 40.0058, "lon": 48.4792},
    "Lankaran":   {"lat": 38.7529, "lon": 48.8518},
    "Nakhchivan": {"lat": 39.2090, "lon": 45.4120}
}

# ── District → Weather Station mapping ───────────────────────────────────────
REGION_TO_WEATHER = {
    "Ganja city":          "Ganja",
    "Goranboy district":   "Ganja",
    "Samukh district":     "Ganja",
    "Goychay district":    "Ganja",
    "Agdash district":     "Ganja",
    "Kurdamir district":   "Ganja",
    "Ujar district":       "Ganja",
    "Yevlakh district":    "Ganja",
    "Zardab district":     "Ganja",
    "Aghsu district":      "Ganja",
    "Aghdara district":    "Ganja",
    "Tartar district":     "Ganja",
    "Aghdam district":     "Ganja",
    "Khojaly district":    "Ganja",
    "Khojavand district":  "Ganja",
    "Fuzuli district":     "Ganja",
    "Sabirabad district":  "Sabirabad",
    "Saatli district":     "Sabirabad",
    "Imishli district":    "Sabirabad",
    "Beylagan district":   "Sabirabad",
    "Aghjabadi district":  "Sabirabad",
    "Barda district":      "Sabirabad",
    "Hajigabul district":  "Sabirabad",
    "Jalilabad district":  "Lankaran",
    "Bilasuvar district":  "Lankaran",
    "Neftchala district":  "Lankaran",
    "Salyan district":     "Lankaran",
    "Jabrayil district":   "Sabirabad",
    "Lachin district":     "Ganja",
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
PREDICT_YEARS     = [2025, 2026] 