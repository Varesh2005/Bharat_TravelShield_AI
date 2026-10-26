'use client';

import React from 'react';
import { Shield, AlertTriangle, Calendar, MapPin, Loader2 } from 'lucide-react';
import HazardCard from './HazardCard';

export default function TravelRiskPanel({
  selectedDistrict,
  travelDate,
  prediction,
  loading,
  error,
}) {
  const hazards = prediction?.hazards || {};
  const hazardKeys = ['malaria', 'heatwave', 'flood', 'dengue'];

  return (
    <div className="flex flex-col h-full bg-slate-50 border border-slate-300 rounded-lg p-4 sm:p-5 shadow-sm">
      {/* Panel Title adhering to wireframe theme */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-200">
        <div>
          <h2 className="text-base sm:text-lg font-bold text-slate-900 flex items-center gap-2">
            <Shield className="w-5 h-5 text-navy-900" />
            Travel Risk Intelligence
          </h2>
          <p className="text-xs text-slate-500">
            Spatiotemporal multi-hazard risk assessment
          </p>
        </div>

        {/* Selected Context Badge */}
        <div className="text-right">
          <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-navy-900 text-white text-xs font-semibold">
            <MapPin className="w-3.5 h-3.5 text-skybtn-400" />
            <span>{selectedDistrict?.district_name || 'Destination'}</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5 flex items-center justify-end gap-1">
            <Calendar className="w-3 h-3" />
            <span>{travelDate || 'Date not set'}</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="mt-4 flex-1 overflow-y-auto space-y-4 pr-1">
        {loading ? (
          <div className="h-64 flex flex-col items-center justify-center gap-3 text-slate-500">
            <Loader2 className="w-8 h-8 animate-spin text-navy-900" />
            <span className="text-xs font-medium">Running ANN hazard inference...</span>
          </div>
        ) : error ? (
          <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600 mt-0.5" />
            <div>
              <span className="font-semibold">Prediction notice: </span>
              {error}
            </div>
          </div>
        ) : null}

        {/* 4 Hazard Risk Cards */}
        <div className="grid grid-cols-1 gap-3.5">
          {hazardKeys.map((key) => (
            <HazardCard key={key} hazardKey={key} cardData={hazards[key]} />
          ))}
        </div>

        {/* Disclaimer strictly adhering to project specifications */}
        <div className="mt-4 p-3 rounded bg-slate-100 border border-slate-200 text-[11px] text-slate-500 leading-relaxed">
          <span className="font-semibold text-slate-700">Disclaimer: </span>
          {prediction?.disclaimer ||
            'Decision support only. Bharat-TravelShield AI predictions are experimental artificial neural network estimates. This is not medical advice, a diagnostic tool, an emergency alert, or a guarantee of travel safety.'}
        </div>
      </div>
    </div>
  );
}
