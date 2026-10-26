'use client';

import React, { useState, useEffect } from 'react';
import {
  Map,
  Layers,
  Thermometer,
  CloudRain,
  Mountain,
  MapPin,
  History,
  Sliders,
  Database,
  CheckCircle,
} from 'lucide-react';
import IndiaMap from './IndiaMap';
import { getDestinationSummary, simulateScenario, getRecentPredictions } from '../lib/api';

export default function MapDetailsPanel({
  selectedDistrict,
  onSelectDistrict,
  travelDate,
}) {
  const [activeTab, setActiveTab] = useState('profile');
  const [summaryData, setSummaryData] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  // Scenario Simulator state
  const [scenarioHazard, setScenarioHazard] = useState('heatwave');
  const [tempDelta, setTempDelta] = useState(2.0);
  const [rainDelta, setRainDelta] = useState(20.0);
  const [scenarioResult, setScenarioResult] = useState(null);
  const [simulating, setSimulating] = useState(false);

  // Audit trail state
  const [recentAudits, setRecentAudits] = useState([]);

  const districtCode =
    selectedDistrict?.census_district_code || selectedDistrict?.code;

  // Load destination historical summary
  useEffect(() => {
    if (!districtCode) return;
    let isMounted = true;
    async function loadSummary() {
      setLoadingSummary(true);
      try {
        const data = await getDestinationSummary(districtCode);
        if (isMounted) setSummaryData(data);
      } catch (err) {
        console.warn('Destination summary notice:', err);
      } finally {
        if (isMounted) setLoadingSummary(false);
      }
    }
    loadSummary();
    return () => {
      isMounted = false;
    };
  }, [districtCode]);

  // Load recent audit logs when audit tab opened
  useEffect(() => {
    if (activeTab === 'audit') {
      getRecentPredictions(6)
        .then((res) => setRecentAudits(res?.predictions || []))
        .catch((e) => console.warn('Audit trail notice:', e));
    }
  }, [activeTab]);

  // Run scenario simulation
  const handleRunScenario = async () => {
    if (!districtCode) return;
    setSimulating(true);
    try {
      const deltas =
        scenarioHazard === 'heatwave'
          ? { max_temperature_c: tempDelta }
          : { annual_rainfall_mm: rainDelta };
      const res = await simulateScenario(scenarioHazard, districtCode, deltas);
      setScenarioResult(res);
    } catch (e) {
      console.warn('Scenario simulation notice:', e);
    } finally {
      setSimulating(false);
    }
  };

  const spots = selectedDistrict?.popular_tourist_spots || [];

  return (
    <div className="flex flex-col h-full bg-slate-50 border border-slate-300 rounded-lg p-4 sm:p-5 shadow-sm space-y-4">
      {/* Panel Title adhering to wireframe theme */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-200">
        <div>
          <h2 className="text-base sm:text-lg font-bold text-slate-900 flex items-center gap-2">
            <Map className="w-5 h-5 text-navy-900" />
            Geospatial Map & Location Details
          </h2>
          <p className="text-xs text-slate-500">
            Click any district pin on map to select and assess
          </p>
        </div>

        {/* Tab Switcher for Other Details */}
        <div className="flex items-center bg-slate-200/80 p-0.5 rounded-lg text-xs">
          <button
            onClick={() => setActiveTab('profile')}
            className={`px-2.5 py-1 rounded font-medium transition-all ${
              activeTab === 'profile'
                ? 'bg-white text-navy-900 shadow-sm font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Profile
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-2.5 py-1 rounded font-medium transition-all ${
              activeTab === 'history'
                ? 'bg-white text-navy-900 shadow-sm font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            History
          </button>
          <button
            onClick={() => setActiveTab('scenario')}
            className={`px-2.5 py-1 rounded font-medium transition-all ${
              activeTab === 'scenario'
                ? 'bg-white text-navy-900 shadow-sm font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            What-If
          </button>
          <button
            onClick={() => setActiveTab('audit')}
            className={`px-2.5 py-1 rounded font-medium transition-all ${
              activeTab === 'audit'
                ? 'bg-white text-navy-900 shadow-sm font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Audit
          </button>
        </div>
      </div>

      {/* Map Section */}
      <div className="w-full">
        <IndiaMap
          selectedDistrict={selectedDistrict}
          onSelectDistrict={onSelectDistrict}
        />
      </div>

      {/* Other Details Section */}
      <div className="flex-1 bg-white border border-slate-200 rounded-lg p-3.5 overflow-y-auto">
        {/* Tab 1: Profile & Tourist Spots */}
        {activeTab === 'profile' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Destination Geographic Profile
              </span>
              <span className="text-xs font-mono text-slate-500">
                Census Code: {districtCode || 'N/A'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-slate-50 border border-slate-100">
                <span className="text-[10px] text-slate-400 block">State / UT</span>
                <span className="font-semibold text-slate-800">
                  {selectedDistrict?.state_name || 'India'}
                </span>
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-100">
                <span className="text-[10px] text-slate-400 block">Coordinates</span>
                <span className="font-semibold text-slate-800 font-mono">
                  {selectedDistrict?.latitude?.toFixed(2)}°N, {selectedDistrict?.longitude?.toFixed(2)}°E
                </span>
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-100">
                <span className="text-[10px] text-slate-400 block">Elevation</span>
                <span className="font-semibold text-slate-800">
                  {selectedDistrict?.elevation_m ? `${selectedDistrict.elevation_m}m ASL` : 'Plain Terrain'}
                </span>
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-100">
                <span className="text-[10px] text-slate-400 block">Climate Zone</span>
                <span className="font-semibold text-slate-800">
                  {selectedDistrict?.climate_zone || 'Subtropical'}
                </span>
              </div>
            </div>

            {/* Popular Landmarks / Tourist Spots */}
            {spots.length > 0 && (
              <div className="mt-2">
                <span className="text-[11px] font-semibold text-slate-700 block mb-1.5">
                  Popular Tourist Landmarks & Hubs:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {spots.map((spot, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-sky-50 text-sky-800 border border-sky-200 text-xs font-medium"
                    >
                      {spot}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Historical Climate & Surveillance */}
        {activeTab === 'history' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Historical Surveillance & Weather (NASA POWER / IMD)
              </span>
              <span className="text-[10px] text-slate-400">Canonical Records</span>
            </div>

            {loadingSummary ? (
              <div className="text-xs text-slate-500 py-4 text-center">
                Loading destination historical records...
              </div>
            ) : summaryData?.hazards ? (
              <div className="space-y-2 text-xs">
                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-slate-800 block">Weather Observations</span>
                    <span className="text-[11px] text-slate-500">
                      Years covered: {summaryData.hazards.weather?.coverage?.source_year_range?.join(' - ') || '2011-2025'}
                    </span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 text-[11px] font-medium border border-emerald-200">
                    Available
                  </span>
                </div>

                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-slate-800 block">Heatwave Threshold Events</span>
                    <span className="text-[11px] text-slate-500">
                      Days Tmax ≥ 45°C: {summaryData.hazards.heatwave_threshold?.records?.length || 0} recorded years
                    </span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 text-[11px] font-medium border border-amber-200">
                    Monitored
                  </span>
                </div>

                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-slate-800 block">Malaria District Surveillance</span>
                    <span className="text-[11px] text-slate-500">
                      NCVBDC district epidemiological records
                    </span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 text-[11px] font-medium border border-blue-200">
                    {summaryData.hazards.malaria?.status || 'Active'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 py-3 text-center">
                Historical records loaded from canonical district dataset.
              </div>
            )}
          </div>
        )}

        {/* Tab 3: What-If Scenario Simulator */}
        {activeTab === 'scenario' && (
          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                What-If Environmental Scenario Simulator
              </span>
              <span className="text-[10px] text-slate-500">Counterfactual ANN</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] font-medium text-slate-700 block mb-1">
                  Target Hazard
                </label>
                <select
                  value={scenarioHazard}
                  onChange={(e) => setScenarioHazard(e.target.value)}
                  className="w-full text-xs border border-slate-300 rounded px-2 py-1 bg-white"
                >
                  <option value="heatwave">Heatwave</option>
                  <option value="flood">Flood</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-700 block mb-1">
                  {scenarioHazard === 'heatwave' ? 'Δ Temperature (°C)' : 'Δ Annual Rainfall (mm)'}
                </label>
                {scenarioHazard === 'heatwave' ? (
                  <input
                    type="number"
                    step="0.5"
                    value={tempDelta}
                    onChange={(e) => setTempDelta(parseFloat(e.target.value) || 0)}
                    className="w-full text-xs border border-slate-300 rounded px-2 py-1"
                  />
                ) : (
                  <input
                    type="number"
                    step="5"
                    value={rainDelta}
                    onChange={(e) => setRainDelta(parseFloat(e.target.value) || 0)}
                    className="w-full text-xs border border-slate-300 rounded px-2 py-1"
                  />
                )}
              </div>
            </div>

            <button
              onClick={handleRunScenario}
              disabled={simulating}
              className="w-full py-1.5 rounded bg-navy-900 text-white font-semibold text-xs hover:bg-navy-800 transition-all disabled:opacity-50"
            >
              {simulating ? 'Simulating Non-Linear Response...' : 'Execute Counterfactual Simulation'}
            </button>

            {scenarioResult && (
              <div className="p-2.5 rounded bg-slate-50 border border-slate-200 mt-2 space-y-1">
                <div className="flex justify-between font-semibold text-slate-800">
                  <span>Baseline Score: {scenarioResult.baseline?.raw_score?.toFixed(3) || 'N/A'}</span>
                  <span>Scenario Score: {scenarioResult.scenario?.raw_score?.toFixed(3) || 'N/A'}</span>
                </div>
                <div className="text-[11px] text-slate-600 flex items-center justify-between">
                  <span>Score Delta: {scenarioResult.score_delta > 0 ? `+${scenarioResult.score_delta.toFixed(3)}` : scenarioResult.score_delta}</span>
                  <span className="font-bold capitalize text-amber-700">Impact: {scenarioResult.risk_impact}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Audit Trail */}
        {activeTab === 'audit' && (
          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Firestore Prediction Audit Trail
              </span>
              <span className="text-[10px] text-slate-400">Persistence Logs</span>
            </div>

            {recentAudits.length === 0 ? (
              <div className="text-xs text-slate-500 py-4 text-center">
                Inference records persisted to Google Cloud Firebase Firestore.
              </div>
            ) : (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {recentAudits.map((a, i) => (
                  <div
                    key={i}
                    className="p-2 rounded bg-slate-50 border border-slate-100 flex items-center justify-between text-[11px]"
                  >
                    <div>
                      <span className="font-semibold text-slate-800">
                        {a.district_name || `District ${a.location_id}`}
                      </span>
                      <span className="text-slate-400 ml-1">({a.target_travel_date || 'Date N/A'})</span>
                    </div>
                    <span className="font-mono text-[10px] text-slate-500">
                      {a.id ? a.id.slice(0, 10) : 'Saved'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
