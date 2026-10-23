"""
Bharat-TravelShield AI - Comprehensive Firebase Firestore Seed Script
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Populates Firestore with:
  1. 32 key Indian tourist and economic hubs with accurate geolocations, elevation, and climate zones.
  2. Baseline historical weather observations (IMD standards).
  3. Baseline epidemiological disease surveillance records (NVBDCP standards).
  4. Registered ML model metadata (MultiHazardRiskANN v1.0).
"""

import os
import sys
from datetime import datetime, timezone

from database.firebase_client import get_firestore_db, is_mock_db
from database.schema import (
    COLLECTION_LOCATIONS,
    COLLECTION_WEATHER,
    COLLECTION_DISEASE,
    COLLECTION_MODELS,
    create_location_doc,
    create_weather_doc,
    create_disease_doc,
)

SAMPLE_DISTRICTS = [
    # Northern Plains & Capital
    {"id": "delhi_central", "district": "Central Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "elev": 216, "climate": "Semi-Arid Subtropical", "pop": 11300, "spots": ["India Gate", "Red Fort", "Qutub Minar"]},
    {"id": "uttar_pradesh_agra", "district": "Agra", "state": "Uttar Pradesh", "lat": 27.1767, "lon": 78.0081, "elev": 171, "climate": "Subtropical", "pop": 1084, "spots": ["Taj Mahal", "Agra Fort", "Fatehpur Sikri"]},
    {"id": "uttar_pradesh_lucknow", "district": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "elev": 123, "climate": "Humid Subtropical", "pop": 1815, "spots": ["Bara Imambara", "Rumi Darwaza"]},
    {"id": "uttar_pradesh_varanasi", "district": "Varanasi", "state": "Uttar Pradesh", "lat": 25.3176, "lon": 82.9739, "elev": 81, "climate": "Humid Subtropical", "pop": 2399, "spots": ["Kashi Vishwanath", "Ghats of Ganga", "Sarnath"]},
    {"id": "punjab_amritsar", "district": "Amritsar", "state": "Punjab", "lat": 31.6340, "lon": 74.8723, "elev": 234, "climate": "Semi-Arid", "pop": 932, "spots": ["Golden Temple", "Wagah Border", "Jallianwala Bagh"]},
    {"id": "chandigarh_chandigarh", "district": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "elev": 321, "climate": "Subtropical", "pop": 9252, "spots": ["Rock Garden", "Sukhna Lake"]},

    # Western & Coastal
    {"id": "maharashtra_mumbai", "district": "Mumbai Suburban", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "elev": 14, "climate": "Tropical Wet & Dry", "pop": 20980, "spots": ["Gateway of India", "Marine Drive", "Elephanta Caves"]},
    {"id": "maharashtra_pune", "district": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "elev": 560, "climate": "Semi-Arid Tropical", "pop": 603, "spots": ["Shaniwar Wada", "Aga Khan Palace", "Sinhagad"]},
    {"id": "goa_north_goa", "district": "North Goa", "state": "Goa", "lat": 15.4989, "lon": 73.8278, "elev": 10, "climate": "Tropical Monsoon", "pop": 471, "spots": ["Calangute Beach", "Fort Aguada", "Anjuna"]},
    {"id": "gujarat_ahmedabad", "district": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "elev": 53, "climate": "Semi-Arid", "pop": 890, "spots": ["Sabarmati Ashram", "Adalaj Stepwell"]},

    # Southern
    {"id": "karnataka_bengaluru", "district": "Bengaluru Urban", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "elev": 920, "climate": "Tropical Savanna", "pop": 4381, "spots": ["Lalbagh", "Cubbon Park", "Bangalore Palace"]},
    {"id": "tamil_nadu_chennai", "district": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "elev": 6, "climate": "Tropical Wet & Dry", "pop": 26903, "spots": ["Marina Beach", "Kapaleeshwarar Temple"]},
    {"id": "kerala_kochi", "district": "Ernakulam (Kochi)", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "elev": 4, "climate": "Tropical Monsoon", "pop": 1072, "spots": ["Fort Kochi", "Chinese Fishing Nets", "Mattancherry"]},
    {"id": "kerala_thiruvananthapuram", "district": "Thiruvananthapuram", "state": "Kerala", "lat": 8.5241, "lon": 76.9366, "elev": 10, "climate": "Tropical Wet & Dry", "pop": 1508, "spots": ["Padmanabhaswamy Temple", "Kovalam Beach"]},
    {"id": "telangana_hyderabad", "district": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "elev": 542, "climate": "Tropical Wet & Dry", "pop": 18172, "spots": ["Charminar", "Golconda Fort", "Hussain Sagar"]},

    # High Altitude & Himalayan
    {"id": "ladakh_leh", "district": "Leh", "state": "Ladakh", "lat": 34.1526, "lon": 77.5771, "elev": 3524, "climate": "Cold Desert", "pop": 3, "spots": ["Pangong Lake", "Nubra Valley", "Thiksey Monastery"]},
    {"id": "jammu_kashmir_srinagar", "district": "Srinagar", "state": "Jammu & Kashmir", "lat": 34.0837, "lon": 74.7973, "elev": 1585, "climate": "Humid Subtropical Montane", "pop": 643, "spots": ["Dal Lake", "Mughal Gardens", "Shankaracharya Temple"]},
    {"id": "himachal_pradesh_shimla", "district": "Shimla", "state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734, "elev": 2276, "climate": "Subtropical Highland", "pop": 159, "spots": ["The Ridge", "Mall Road", "Jakhu Temple"]},
    {"id": "himachal_pradesh_kullu", "district": "Kullu (Manali)", "state": "Himachal Pradesh", "lat": 31.9579, "lon": 77.1095, "elev": 2050, "climate": "Subtropical Highland", "pop": 79, "spots": ["Solang Valley", "Rohtang Pass", "Hadimba Temple"]},
    {"id": "sikkim_gangtok", "district": "East Sikkim (Gangtok)", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065, "elev": 1650, "climate": "Subtropical Highland", "pop": 297, "spots": ["Tsomgo Lake", "Rumtek Monastery", "Nathula Pass"]},

    # Eastern & North-Eastern
    {"id": "west_bengal_kolkata", "district": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "elev": 9, "climate": "Tropical Wet & Dry", "pop": 24252, "spots": ["Victoria Memorial", "Howrah Bridge", "Dakshineswar"]},
    {"id": "assam_kamrup", "district": "Kamrup Metropolitan (Guwahati)", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "elev": 55, "climate": "Humid Subtropical", "pop": 2010, "spots": ["Kamakhya Temple", "Umananda Island", "Brahmaputra Cruise"]},
    {"id": "meghalaya_shillong", "district": "East Khasi Hills (Shillong)", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933, "elev": 1525, "climate": "Subtropical Highland", "pop": 292, "spots": ["Elephant Falls", "Shillong Peak", "Umiam Lake"]},
    {"id": "odisha_bhubaneswar", "district": "Khordha (Bhubaneswar)", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "elev": 45, "climate": "Tropical Savanna", "pop": 800, "spots": ["Lingaraj Temple", "Udayagiri Caves", "Puri Jagannath nearby"]},
    {"id": "bihar_patna", "district": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "elev": 53, "climate": "Humid Subtropical", "pop": 1823, "spots": ["Golghar", "Patna Museum", "Takht Sri Patna Sahib"]},

    # Desert & Central
    {"id": "rajasthan_jaipur", "district": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "elev": 431, "climate": "Semi-Arid", "pop": 598, "spots": ["Hawa Mahal", "Amber Fort", "City Palace"]},
    {"id": "rajasthan_jaisalmer", "district": "Jaisalmer", "state": "Rajasthan", "lat": 26.9157, "lon": 70.9083, "elev": 225, "climate": "Arid Desert", "pop": 17, "spots": ["Jaisalmer Fort", "Sam Sand Dunes", "Patwon Ki Haveli"]},
    {"id": "madhya_pradesh_bhopal", "district": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "elev": 527, "climate": "Humid Subtropical", "pop": 855, "spots": ["Upper Lake", "Van Vihar", "Sanchi Stupa nearby"]},
    {"id": "madhya_pradesh_indore", "district": "Indore", "state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577, "elev": 553, "climate": "Tropical Wet & Dry", "pop": 841, "spots": ["Rajwada", "Lal Bagh Palace", "Sarafa Bazaar"]},
]


def seed_database():
    """Seeds Firebase Firestore collections with comprehensive foundation data."""
    print("=== Bharat-TravelShield AI — Seeding Firebase Firestore ===")
    db = get_firestore_db()
    print(f"Firestore Client Status: {'Mock Memory Mode' if is_mock_db() else 'Connected to Google Cloud'}")

    loc_coll = db.collection(COLLECTION_LOCATIONS)
    weather_coll = db.collection(COLLECTION_WEATHER)
    disease_coll = db.collection(COLLECTION_DISEASE)
    model_coll = db.collection(COLLECTION_MODELS)

    # 1. Seed Locations (using canonical district_master.csv if available, else sample)
    master_csv = os.path.join("data", "reference", "district_master.csv")
    if os.path.exists(master_csv):
        import csv
        print(f"Loading canonical district master from {master_csv}...")
        count = 0
        with open(master_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("census_district_code", "")
                dist_name = row.get("district_name", "")
                st_name = row.get("state_name", "")
                doc_id = f"{st_name}_{dist_name}".lower().replace(" ", "_").replace("&", "and")
                doc = {
                    "id": doc_id,
                    "census_district_code": int(code) if code and code.isdigit() else None,
                    "district_name": dist_name,
                    "state_name": st_name,
                    "normalized_key": f"{st_name}_{dist_name}".lower().replace(" ", "_"),
                    "latitude": float(row.get("latitude", 20.0)),
                    "longitude": float(row.get("longitude", 78.0)),
                    "elevation_m": 200.0,
                    "climate_zone": "Subtropical",
                    "population_density": float(row.get("population_2011", 500000)) / 1000.0,
                    "popular_tourist_spots": [],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                loc_coll.document(doc_id).set(doc, merge=True)
                count += 1
        print(f"Successfully seeded {count} canonical districts from district_master.csv!")
    else:
        print(f"Seeding {len(SAMPLE_DISTRICTS)} sample Indian districts...")
        for dist in SAMPLE_DISTRICTS:
            doc = create_location_doc(
                location_id=dist["id"],
                district_name=dist["district"],
                state_name=dist["state"],
                latitude=dist["lat"],
                longitude=dist["lon"],
                elevation_m=dist["elev"],
                climate_zone=dist["climate"],
                population_density=dist["pop"],
                popular_tourist_spots=dist["spots"],
            )
            loc_coll.document(dist["id"]).set(doc, merge=True)
        print("Successfully seeded sample locations!")

    # 2. Seed Baseline Weather Observations for Key Hubs
    print("Seeding baseline historical weather observations...")
    weather_samples = [
        {"loc": "delhi_central", "date": "2026-07-15", "max_t": 35.2, "min_t": 27.4, "r": 88.0, "rh": 82.0},
        {"loc": "delhi_central", "date": "2026-05-20", "max_t": 43.8, "min_t": 29.5, "r": 0.0, "rh": 28.0},
        {"loc": "maharashtra_mumbai", "date": "2026-07-20", "max_t": 30.5, "min_t": 25.1, "r": 145.0, "rh": 91.0},
        {"loc": "kerala_kochi", "date": "2026-06-18", "max_t": 29.8, "min_t": 24.2, "r": 160.0, "rh": 94.0},
        {"loc": "rajasthan_jaipur", "date": "2026-05-25", "max_t": 44.5, "min_t": 30.1, "r": 0.0, "rh": 22.0},
        {"loc": "ladakh_leh", "date": "2026-07-10", "max_t": 24.0, "min_t": 11.0, "r": 4.0, "rh": 35.0},
    ]

    for w in weather_samples:
        doc = create_weather_doc(
            location_id=w["loc"],
            observation_date=w["date"],
            max_temp_c=w["max_t"],
            min_temp_c=w["min_t"],
            rainfall_mm=w["r"],
            humidity_pct=w["rh"],
            source="IMD",
        )
        weather_coll.document(f"{w['loc']}_{w['date']}").set(doc, merge=True)

    # 3. Seed Baseline Disease Records
    print("Seeding baseline epidemiological surveillance records...")
    disease_samples = [
        {"loc": "delhi_central", "date": "2026-08-01", "type": "Dengue", "cases": 142},
        {"loc": "delhi_central", "date": "2026-08-01", "type": "Malaria", "cases": 28},
        {"loc": "maharashtra_mumbai", "date": "2026-08-01", "type": "Dengue", "cases": 210},
        {"loc": "kerala_kochi", "date": "2026-08-01", "type": "Dengue", "cases": 95},
        {"loc": "assam_kamrup", "date": "2026-07-15", "type": "Malaria", "cases": 74},
    ]

    for d in disease_samples:
        doc = create_disease_doc(
            location_id=d["loc"],
            report_date=d["date"],
            disease_type=d["type"],
            cases_reported=d["cases"],
            source="NVBDCP",
        )
        disease_coll.document(f"{d['loc']}_{d['type']}_{d['date']}").set(doc, merge=True)

    # 4. Register ML Model in Database
    print("Registering MultiHazardRiskANN in Firestore 'ml_models'...")
    model_doc = {
        "model_id": "MultiHazardRiskANN-v1.0",
        "name": "Multi-Hazard Spatiotemporal Risk Neural Network",
        "architecture": "Multi-Task MLP (Shared Trunk + 4 Specialized Hazard Heads)",
        "author": "Varesh (ML/ANN) & Pranav (DB/Integration)",
        "framework": "PyTorch 2.14 / NumPy",
        "inputs_dim": 18,
        "hazards": ["Dengue", "Malaria", "Flood", "Heatwave"],
        "metrics": {
            "dengue_roc_auc": 0.9130,
            "malaria_roc_auc": 0.8852,
            "flood_roc_auc": 0.9368,
            "heatwave_roc_auc": 0.9736,
        },
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True,
    }
    model_coll.document("MultiHazardRiskANN-v1.0").set(model_doc, merge=True)

    print("=== Firebase Firestore Seeding Successfully Completed! ===")


if __name__ == "__main__":
    seed_database()
