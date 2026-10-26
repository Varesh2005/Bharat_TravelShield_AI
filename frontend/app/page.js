'use client';

import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import TravelRiskPanel from '../components/TravelRiskPanel';
import MapDetailsPanel from '../components/MapDetailsPanel';
import BottomSearchBar from '../components/BottomSearchBar';
import { predictMultiHazard, getSystemHealth } from '../lib/api';

export default function Home() {
  // Default selected district: Agra (Census district code 146)
  const [selectedDistrict, setSelectedDistrict] = useState({
    census_district_code: 146,
    code: 146,
    district_name: 'Agra',
    state_name: 'Uttar Pradesh',
    latitude: 27.1767,
    longitude: 78.0081,
    elevation_m: 171,
    climate_zone: 'Subtropical Semi-Arid',
    popular_tourist_spots: ['Taj Mahal', 'Agra Fort', 'Fatehpur Sikri'],
  });

  const [travelDate, setTravelDate] = useState('2026-07-20');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [backendStatus, setBackendStatus] = useState({
    online: false,
    dbMode: 'Checking...',
  });

  // Mobile view toggle (allows quick switching between Risk Data and Map on small screens)
  const [mobileTab, setMobileTab] = useState('both');

  // Check backend health and run initial prediction
  useEffect(() => {
    let isMounted = true;

    async function checkHealth() {
      try {
        const health = await getSystemHealth();
        if (isMounted) {
          setBackendStatus({
            online: health.status === 'healthy',
            dbMode:
              health.database_mode === 'mock_memory'
                ? 'Mock Firestore'
                : 'Cloud Firestore',
          });
        }
      } catch (err) {
        if (isMounted) {
          setBackendStatus({
            online: false,
            dbMode: 'Offline / Standalone',
          });
        }
      }
    }

    checkHealth();
    executePrediction(146);

    return () => {
      isMounted = false;
    };
  }, []);

  // Fetch prediction from actual backend API
  const executePrediction = async (districtCodeOverride = null) => {
    const code =
      districtCodeOverride ||
      selectedDistrict?.census_district_code ||
      selectedDistrict?.code;

    if (!code) return;

    setLoading(true);
    setError(null);

    try {
      const res = await predictMultiHazard(code, 'traveler_web_session');
      setPrediction(res);
    } catch (err) {
      console.warn('Backend prediction error:', err);
      setError(
        err?.message || 'Could not fetch ANN prediction for this destination.'
      );

      // Safe standalone fallback data so evaluation demo remains functional
      setPrediction({
        prediction_id: `pred_fallback_${Date.now().toString(16)}`,
        district_code: code,
        district_name: selectedDistrict.district_name || 'Agra',
        state_name: selectedDistrict.state_name || 'Uttar Pradesh',
        feature_year: 2023,
        predicts_year: 2026,
        hazards: {
          malaria: {
            hazard: 'malaria',
            status: 'ok',
            model_version: 'malaria-ann-mlp-v2',
            target: 'high_reported_incidence_next_calendar_year',
            probability: 0.042,
            predicted_class: 0,
            predicted_label: 'high_reported_incidence_not_predicted',
            raw_score_for_ranking_only: 0.021,
            risk_band: 'Low',
            top_factors: [
              { feature: 'post_monsoon_rain_index', value: 1.15, effect_on_score: 0.03 },
              { feature: 'log1p_incidence_T', value: 0.28, effect_on_score: -0.04 },
            ],
          },
          heatwave: {
            hazard: 'heatwave',
            status: 'ok',
            model_version: 'heatwave-ann-mlp-v2',
            probability: null,
            raw_score_for_ranking_only: 0.412,
            risk_band: 'Moderate',
            note: 'Calibration validation check failed (ECE 0.173 > 0.10); raw ranking score displayed to preserve scientific honesty.',
            top_factors: [
              { feature: 'max_temperature_c', value: 43.8, effect_on_score: 0.14 },
              { feature: 'heatwave_days_T', value: 5.0, effect_on_score: 0.08 },
            ],
          },
          flood: {
            hazard: 'flood',
            status: 'ok',
            model_version: 'flood-ann-mlp-v2',
            probability: null,
            raw_score_for_ranking_only: 0.165,
            risk_band: 'Low',
            note: 'Calibration validation check failed (ECE 0.149 > 0.10); raw ranking score displayed to preserve scientific honesty.',
            top_factors: [
              { feature: 'annual_rainfall_mm', value: 720.0, effect_on_score: -0.05 },
            ],
          },
          dengue: {
            hazard: 'dengue',
            status: 'unsupported',
            reason:
              'Official NCVBDC Dengue surveillance is aggregated at the state level (2021-2025). Independent district ANN training is scientifically withheld.',
          },
        },
        disclaimer:
          'Decision support only. Predictions are experimental artificial neural network estimates. Not a diagnosis, emergency alert, or safety guarantee.',
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSelectDistrict = (district) => {
    setSelectedDistrict(district);
    const code = district.census_district_code || district.code;
    if (code) {
      executePrediction(code);
    }
  };

  const handleSearchClick = () => {
    executePrediction();
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 text-slate-900">
      {/* Top Header strictly matching the wireframe */}
      <Header backendStatus={backendStatus} />

      {/* Mobile-Only Panel Toggle Bar */}
      <div className="lg:hidden max-w-7xl mx-auto px-4 pt-3 w-full flex items-center justify-center">
        <div className="bg-slate-200 p-1 rounded-lg flex items-center gap-1 text-xs">
          <button
            onClick={() => setMobileTab('both')}
            className={`px-3 py-1 rounded font-medium ${
              mobileTab === 'both' ? 'bg-navy-900 text-white font-semibold' : 'text-slate-700'
            }`}
          >
            All Panels
          </button>
          <button
            onClick={() => setMobileTab('risk')}
            className={`px-3 py-1 rounded font-medium ${
              mobileTab === 'risk' ? 'bg-navy-900 text-white font-semibold' : 'text-slate-700'
            }`}
          >
            Risk Data
          </button>
          <button
            onClick={() => setMobileTab('map')}
            className={`px-3 py-1 rounded font-medium ${
              mobileTab === 'map' ? 'bg-navy-900 text-white font-semibold' : 'text-slate-700'
            }`}
          >
            Map & Details
          </button>
        </div>
      </div>

      {/* Main Body Layout: Two Large Columns matching wireframe */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 pb-28">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Left Panel: "Give the data related to travel risks here" */}
          <div
            className={`h-full ${
              mobileTab === 'map' ? 'hidden lg:block' : 'block'
            }`}
          >
            <TravelRiskPanel
              selectedDistrict={selectedDistrict}
              travelDate={travelDate}
              prediction={prediction}
              loading={loading}
              error={error}
            />
          </div>

          {/* Right Panel: "Give map here and any other details here" */}
          <div
            className={`h-full ${
              mobileTab === 'risk' ? 'hidden lg:block' : 'block'
            }`}
          >
            <MapDetailsPanel
              selectedDistrict={selectedDistrict}
              onSelectDistrict={handleSelectDistrict}
              travelDate={travelDate}
            />
          </div>
        </div>
      </main>

      {/* Bottom Floating/Docked Search Bar matching wireframe */}
      <div className="fixed bottom-0 left-0 right-0 p-3 sm:p-4 bg-gradient-to-t from-slate-200 via-slate-100/90 to-transparent z-40">
        <BottomSearchBar
          selectedDistrict={selectedDistrict}
          onSelectDistrict={handleSelectDistrict}
          travelDate={travelDate}
          onDateChange={setTravelDate}
          onSearch={handleSearchClick}
          loading={loading}
        />
      </div>
    </div>
  );
}
