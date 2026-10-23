"""
Bharat-TravelShield AI - Map Layers Service
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Provides lightweight spatial data for MapLibre GL choropleth visualization.
"""

import os
import csv
from typing import Dict, Any, List


class MapService:
    """Provides district-level risk mapping for frontend maps."""

    @staticmethod
    def get_hazard_layer(hazard_type: str) -> Dict[str, Any]:
        """
        Returns lightweight district hazard data dictionary for MapLibre GL.
        """
        hz = hazard_type.lower().strip()
        master_csv = os.path.join("data", "reference", "district_master.csv")

        if not os.path.exists(master_csv):
            return {"hazard": hz, "count": 0, "districts": []}

        districts = []
        with open(master_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code_str = row.get("census_district_code", "")
                if not code_str:
                    continue
                code = int(code_str)
                dist_name = row.get("district_name", "")
                state_name = row.get("state_name", "")
                lat = float(row.get("latitude", 20.0))
                lon = float(row.get("longitude", 78.0))

                districts.append({
                    "code": code,
                    "district": dist_name,
                    "state": state_name,
                    "lat": lat,
                    "lon": lon,
                })

        return {
            "hazard": hz,
            "count": len(districts),
            "districts": districts,
        }
