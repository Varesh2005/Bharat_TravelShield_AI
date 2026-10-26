'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Map, MapPin, Layers, Compass, Loader2 } from 'lucide-react';
import { getMapLayer } from '../lib/api';

export default function IndiaMap({ selectedDistrict, onSelectDistrict }) {
  const [hazardLayer, setHazardLayer] = useState('heatwave');
  const [districtPoints, setDistrictPoints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Fetch actual district coordinates from backend map service
  useEffect(() => {
    let isMounted = true;
    async function loadDistricts() {
      setLoading(true);
      try {
        const data = await getMapLayer(hazardLayer);
        if (isMounted && data?.districts && data.districts.length > 0) {
          setDistrictPoints(data.districts);
        }
      } catch (err) {
        console.warn('Map layer fetch warning:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadDistricts();
    return () => {
      isMounted = false;
    };
  }, [hazardLayer]);

  // Project actual lat/lon coordinates to SVG viewBox (0 0 800 800)
  // India geographic bounds: ~7.0°N to 37.5°N, 68.0°E to 97.5°E
  const project = (lat, lon) => {
    const minLat = 7.0;
    const maxLat = 37.5;
    const minLon = 68.0;
    const maxLon = 97.5;

    const x = ((lon - minLon) / (maxLon - minLon)) * 680 + 60;
    const y = ((maxLat - lat) / (maxLat - minLat)) * 680 + 60;
    return { x, y };
  };

  const selectedPoint = useMemo(() => {
    if (!selectedDistrict?.latitude || !selectedDistrict?.longitude) {
      // Find matching district in districtPoints
      const code = selectedDistrict?.census_district_code || selectedDistrict?.code;
      if (code) {
        return districtPoints.find((d) => d.code === Number(code));
      }
      return null;
    }
    return {
      lat: selectedDistrict.latitude,
      lon: selectedDistrict.longitude,
      district: selectedDistrict.district_name,
      state: selectedDistrict.state_name,
      code: selectedDistrict.census_district_code || selectedDistrict.code,
    };
  }, [selectedDistrict, districtPoints]);

  const selectedProj = selectedPoint ? project(selectedPoint.lat, selectedPoint.lon) : null;

  return (
    <div className="relative w-full rounded-lg bg-slate-900 border border-slate-700 overflow-hidden text-white flex flex-col">
      {/* Top Map Controls */}
      <div className="p-3 bg-slate-950/80 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2 z-10">
        <div className="flex items-center gap-2">
          <Map className="w-4 h-4 text-skybtn-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
            India Geospatial Layer
          </span>
          <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono">
            {districtPoints.length > 0 ? `${districtPoints.length} Districts` : 'Loading...'}
          </span>
        </div>

        {/* Hazard Layer Switch */}
        <div className="flex items-center gap-1 text-[11px]">
          <span className="text-slate-400 text-[10px] hidden sm:inline mr-1">Layer:</span>
          {['heatwave', 'flood', 'malaria'].map((h) => (
            <button
              key={h}
              onClick={() => setHazardLayer(h)}
              className={`px-2 py-0.5 rounded font-medium capitalize transition-all ${
                hazardLayer === h
                  ? 'bg-skybtn-500 text-navy-950 font-bold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Canvas with Actual District Coordinates */}
      <div className="relative w-full h-[320px] sm:h-[380px] bg-[#070d1e] flex items-center justify-center overflow-hidden">
        {loading && districtPoints.length === 0 ? (
          <div className="flex flex-col items-center gap-2 text-slate-400 text-xs">
            <Loader2 className="w-6 h-6 animate-spin text-skybtn-400" />
            <span>Plotting 640 district coordinates...</span>
          </div>
        ) : (
          <svg
            viewBox="0 0 800 800"
            className="w-full h-full select-none"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Grid references */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#132347" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="800" height="800" fill="url(#grid)" />

            {/* Render actual district coordinate dots */}
            {districtPoints.map((pt) => {
              const { x, y } = project(pt.lat, pt.lon);
              const isSelected = selectedPoint && selectedPoint.code === pt.code;

              return (
                <circle
                  key={pt.code}
                  cx={x}
                  cy={y}
                  r={isSelected ? 6 : 2.5}
                  className={`cursor-pointer transition-all ${
                    isSelected
                      ? 'fill-skybtn-400 stroke-white stroke-2'
                      : 'fill-slate-400 hover:fill-amber-400 hover:r-4'
                  }`}
                  onClick={() =>
                    onSelectDistrict({
                      census_district_code: pt.code,
                      code: pt.code,
                      district_name: pt.district,
                      state_name: pt.state,
                      latitude: pt.lat,
                      longitude: pt.lon,
                    })
                  }
                  onMouseEnter={() => setHoveredPoint(pt)}
                  onMouseLeave={() => setHoveredPoint(null)}
                />
              );
            })}

            {/* Prominent pulse ring around selected destination */}
            {selectedProj && (
              <g>
                <circle
                  cx={selectedProj.x}
                  cy={selectedProj.y}
                  r="16"
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.5"
                  className="animate-ping opacity-75 origin-center"
                />
                <circle
                  cx={selectedProj.x}
                  cy={selectedProj.y}
                  r="9"
                  fill="#00a8e8"
                  stroke="#ffffff"
                  strokeWidth="2"
                />
                {/* District Pin Label */}
                <rect
                  x={selectedProj.x + 12}
                  y={selectedProj.y - 14}
                  width={selectedPoint.district.length * 8 + 18}
                  height="22"
                  rx="4"
                  fill="#081b4b"
                  stroke="#38bdf8"
                  strokeWidth="1"
                />
                <text
                  x={selectedProj.x + 18}
                  y={selectedProj.y + 1}
                  fill="#ffffff"
                  fontSize="11"
                  fontWeight="bold"
                  fontFamily="sans-serif"
                >
                  {selectedPoint.district}
                </text>
              </g>
            )}
          </svg>
        )}

        {/* Hover Tooltip */}
        {hoveredPoint && (
          <div className="absolute bottom-3 left-3 bg-slate-900/90 border border-slate-700 px-3 py-1.5 rounded text-xs shadow-lg pointer-events-none">
            <span className="font-bold text-white">{hoveredPoint.district}</span>
            <span className="text-slate-400 ml-1">({hoveredPoint.state})</span>
            <div className="text-[10px] text-skybtn-400 font-mono">
              Lat: {hoveredPoint.lat.toFixed(2)}°, Lon: {hoveredPoint.lon.toFixed(2)}°
            </div>
          </div>
        )}

        {/* Coordinate Legend */}
        <div className="absolute top-3 right-3 bg-slate-950/80 px-2 py-1 rounded text-[10px] font-mono text-slate-400 border border-slate-800">
          Lat 7.0°N - 37.5°N | Lon 68.0°E - 97.5°E
        </div>
      </div>
    </div>
  );
}
