"""
Bharat-TravelShield AI - Spatiotemporal Feature Preprocessing Pipeline
Author: Varesh (ML / ANN & Research) + Pranav
Project: Bharat-TravelShield AI

This module extracts, normalizes, and packages spatiotemporal features
into a standardized 18-dimensional vector for the MultiHazardRiskANN.
"""

import math
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

from ml.models.ann_model import FEATURE_NAMES, NUM_FEATURES


@dataclass
class SpatiotemporalInput:
    """Input payload representing spatiotemporal context for a district and travel date."""
    target_date: Union[str, datetime]  # YYYY-MM-DD
    latitude: float
    longitude: float
    elevation_m: float = 200.0
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    rainfall_7d_mm: Optional[float] = None
    rainfall_30d_mm: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    dengue_lag_cases: Optional[float] = 10.0
    malaria_lag_cases: Optional[float] = 5.0
    flood_history_freq: Optional[float] = 0.2
    heatwave_history_freq: Optional[float] = 0.2
    population_density: Optional[float] = 800.0


class FeaturePipeline:
    """
    Leakage-safe feature extraction and normalization pipeline for Bharat-TravelShield AI.
    Converts raw inputs into normalized tensors matching the ANN input distribution.
    """

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.feature_dim = NUM_FEATURES

    @staticmethod
    def _parse_month(date_input: Union[str, datetime]) -> int:
        if isinstance(date_input, datetime):
            return date_input.month
        try:
            dt = datetime.strptime(str(date_input)[:10], "%Y-%m-%d")
            return dt.month
        except Exception:
            return 7  # Default to July (monsoon peak) if unparseable

    @staticmethod
    def _get_seasonal_climatology_defaults(month: int, latitude: float) -> Dict[str, float]:
        """
        Provides realistic climatological priors for India if real-time observations
        are missing during a forward query.
        """
        is_northern = latitude > 22.0
        is_monsoon = month in [6, 7, 8, 9]
        is_summer = month in [4, 5]
        is_winter = month in [12, 1, 2]

        if is_monsoon:
            max_temp = 31.0 if is_northern else 29.0
            min_temp = 24.0 if is_northern else 23.0
            r7 = 85.0
            r30 = 260.0
            rh = 82.0
        elif is_summer:
            max_temp = 42.0 if is_northern else 36.0
            min_temp = 27.0 if is_northern else 26.0
            r7 = 5.0
            r30 = 20.0
            rh = 35.0 if is_northern else 60.0
        elif is_winter:
            max_temp = 21.0 if is_northern else 28.0
            min_temp = 9.0 if is_northern else 19.0
            r7 = 2.0
            r30 = 10.0
            rh = 55.0
        else:  # Post-monsoon / Spring
            max_temp = 32.0
            min_temp = 20.0
            r7 = 15.0
            r30 = 50.0
            rh = 65.0

        return {
            "max_temp": max_temp,
            "min_temp": min_temp,
            "rainfall_7d": r7,
            "rainfall_30d": r30,
            "humidity": rh,
        }

    def transform(self, inp: SpatiotemporalInput) -> np.ndarray:
        """
        Transforms SpatiotemporalInput into a 18-element normalized float32 array.
        """
        month = self._parse_month(inp.target_date)

        # 1. Cyclical Month Encoding (ensures Dec 12 and Jan 1 are close in latent space)
        month_rad = 2.0 * math.pi * (month - 1) / 12.0
        month_sin = float(math.sin(month_rad))
        month_cos = float(math.cos(month_rad))

        # 2. Geographic Normalization (Centered roughly on central India: Lat 20°N, Lon 80°E)
        lat_norm = float((inp.latitude - 20.0) / 10.0)
        lon_norm = float((inp.longitude - 80.0) / 10.0)
        elev_norm = float(inp.elevation_m / 1000.0)

        # 3. Climate & Meteorological Features (using domain fallback if null)
        clim_defaults = self._get_seasonal_climatology_defaults(month, inp.latitude)
        max_t = inp.max_temp_c if inp.max_temp_c is not None else clim_defaults["max_temp"]
        min_t = inp.min_temp_c if inp.min_temp_c is not None else clim_defaults["min_temp"]
        r7 = inp.rainfall_7d_mm if inp.rainfall_7d_mm is not None else clim_defaults["rainfall_7d"]
        r30 = inp.rainfall_30d_mm if inp.rainfall_30d_mm is not None else clim_defaults["rainfall_30d"]
        rh = inp.relative_humidity_pct if inp.relative_humidity_pct is not None else clim_defaults["humidity"]

        max_temp_norm = float((max_t - 30.0) / 10.0)
        min_temp_norm = float((min_t - 20.0) / 10.0)
        diurnal_range_norm = float(max(0.0, max_t - min_t) / 10.0)
        rainfall_7d_norm = float(r7 / 100.0)
        rainfall_30d_norm = float(r30 / 300.0)
        humidity_norm = float(rh / 100.0)

        # 4. Seasonal Indicators
        monsoon_active_flag = 1.0 if month in [6, 7, 8, 9] else 0.0
        summer_heat_flag = 1.0 if (month in [4, 5, 6] and max_t >= 38.0) else 0.0

        # 5. Epidemiology & Historical Disaster Priors
        dengue_cases = inp.dengue_lag_cases if inp.dengue_lag_cases is not None else 10.0
        malaria_cases = inp.malaria_lag_cases if inp.malaria_lag_cases is not None else 5.0
        dengue_lag_norm = float(math.log1p(max(0.0, dengue_cases)) / 5.0)
        malaria_lag_norm = float(math.log1p(max(0.0, malaria_cases)) / 5.0)

        flood_freq = float(inp.flood_history_freq if inp.flood_history_freq is not None else 0.2)
        heatwave_freq = float(inp.heatwave_history_freq if inp.heatwave_history_freq is not None else 0.2)

        pop_density = inp.population_density if inp.population_density is not None else 800.0
        pop_density_norm = float(math.log1p(max(0.0, pop_density)) / 10.0)

        vector = np.array([
            month_sin,
            month_cos,
            lat_norm,
            lon_norm,
            elev_norm,
            max_temp_norm,
            min_temp_norm,
            diurnal_range_norm,
            rainfall_7d_norm,
            rainfall_30d_norm,
            humidity_norm,
            monsoon_active_flag,
            summer_heat_flag,
            dengue_lag_norm,
            malaria_lag_norm,
            flood_freq,
            heatwave_freq,
            pop_density_norm,
        ], dtype=np.float32)

        return vector

    def get_feature_dict(self, vector: np.ndarray) -> Dict[str, float]:
        """Maps vector back to named feature dictionary for explainability."""
        return {name: float(vector[i]) for i, name in enumerate(self.feature_names)}
