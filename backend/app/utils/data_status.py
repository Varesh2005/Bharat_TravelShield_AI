"""
Bharat-TravelShield AI - Data Source Audit Utility
Author: Rajdeep (Backend / API)
"""

import os
from typing import Dict, Any


def get_data_sources_status() -> Dict[str, Any]:
    """Inspects canonical and processed dataset files and reports availability."""
    sources = [
        {
            "name": "District Master Reference",
            "type": "geography",
            "file": "data/reference/district_master.csv",
            "provider": "Census 2011 & Survey of India",
        },
        {
            "name": "Weather District Observations",
            "type": "weather",
            "file": "data/processed/weather/weather_district_year_2011_2025.csv",
            "provider": "NASA POWER / IMD",
        },
        {
            "name": "Heatwave Threshold Records",
            "type": "disaster",
            "file": "data/processed/disaster/heatwave/heatwave_district_year_2011_2025.csv",
            "provider": "NASA POWER Tmax >= 45C",
        },
        {
            "name": "Flood Inventory Events",
            "type": "disaster",
            "file": "data/processed/disaster/flood/flood_district_year_1967_2023.csv",
            "provider": "India Flood Inventory v3",
        },
        {
            "name": "Malaria Surveillance Reports",
            "type": "disease",
            "file": "data/processed/disease/malaria/malaria_district_year_2000_2024.csv",
            "provider": "NCVBDC",
        },
        {
            "name": "Dengue Surveillance Context",
            "type": "disease",
            "file": "data/processed/disease/dengue/dengue_state_year_2021_2026.csv",
            "provider": "NCVBDC State Tables",
        },
    ]

    status_report = []
    all_available = True

    for s in sources:
        exists = os.path.exists(s["file"])
        size_bytes = os.path.getsize(s["file"]) if exists else 0
        if not exists:
            all_available = False
        status_report.append({
            "name": s["name"],
            "type": s["type"],
            "provider": s["provider"],
            "status": "available" if exists else "missing",
            "file_path": s["file"],
            "size_kb": round(size_bytes / 1024, 1) if exists else 0,
        })

    return {
        "overall_status": "ready" if all_available else "degraded",
        "sources_count": len(sources),
        "available_count": sum(1 for s in status_report if s["status"] == "available"),
        "sources": status_report,
    }
