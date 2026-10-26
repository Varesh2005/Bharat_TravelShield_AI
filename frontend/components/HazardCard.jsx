'use client';

import React, { useState } from 'react';
import {
  Bug,
  Flame,
  Waves,
  ShieldAlert,
  Info,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
} from 'lucide-react';
import { HAZARD_CONFIG, RISK_TIERS } from '../lib/constants';

const ICON_MAP = {
  malaria: Bug,
  heatwave: Flame,
  flood: Waves,
  dengue: ShieldAlert,
};

export default function HazardCard({ hazardKey, cardData }) {
  const [expanded, setExpanded] = useState(false);
  const config = HAZARD_CONFIG[hazardKey] || {
    name: hazardKey.toUpperCase(),
    badgeColor: 'border-slate-300 bg-slate-50 text-slate-700',
    description: 'Hazard Risk Intelligence',
    targetLabel: 'Standardized Hazard Risk',
    advice: ['Check local administration guidelines.'],
  };

  const Icon = ICON_MAP[hazardKey] || ShieldAlert;

  // Dengue state surveillance case
  if (!cardData || cardData.status === 'unsupported') {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm hover:border-slate-300 transition-all">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded bg-rose-50 border border-rose-200 text-rose-600">
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900">{config.name}</h3>
              <p className="text-xs text-slate-500">State Surveillance Alert</p>
            </div>
          </div>
          <span className="px-2 py-0.5 rounded text-[11px] font-medium border border-rose-200 bg-rose-50 text-rose-700">
            State Advisory
          </span>
        </div>

        <div className="mt-3 p-2.5 rounded bg-slate-50 border border-slate-200 text-xs text-slate-600">
          <div className="flex items-center gap-1.5 font-medium text-slate-700 mb-1">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span>Scientific Data Governance Notice</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-500">
            {cardData?.reason ||
              'Official NCVBDC Dengue surveillance is aggregated at the state level (2021-2025). Independent district ANN training is scientifically withheld.'}
          </p>
        </div>

        <div className="mt-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center justify-between w-full text-xs font-medium text-slate-600 hover:text-slate-900"
          >
            <span>Precautions & Guidance ({config.advice.length})</span>
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {expanded && (
            <ul className="mt-2 space-y-1.5 text-xs text-slate-600 border-t border-slate-100 pt-2">
              {config.advice.map((item, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-1.5 shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    );
  }

  // Value formatting according to academic calibration integrity rules
  const hasCalibratedProb = cardData.probability !== null && cardData.probability !== undefined;
  const rawScore = cardData.raw_score_for_ranking_only;
  const scoreDisplay = hasCalibratedProb
    ? `${(cardData.probability * 100).toFixed(1)}%`
    : rawScore !== null && rawScore !== undefined
    ? rawScore.toFixed(3)
    : 'N/A';

  const scoreLabel = hasCalibratedProb ? 'Calibrated Probability' : 'Raw Ranking Score';

  // Determine risk band
  let riskBand = cardData.risk_band || 'Moderate';
  const tierConfig = RISK_TIERS[riskBand] || RISK_TIERS.Moderate;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm hover:border-slate-300 transition-all">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded bg-slate-100 border border-slate-200 text-slate-700">
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">{config.name}</h3>
            <p className="text-xs text-slate-500">{config.targetLabel}</p>
          </div>
        </div>
        <span className={`px-2 py-0.5 rounded text-[11px] font-medium border ${tierConfig.badge}`}>
          {riskBand}
        </span>
      </div>

      {/* Main Score Display */}
      <div className="mt-3 flex items-baseline justify-between border-b border-slate-100 pb-2.5">
        <div>
          <span className="text-2xl font-bold text-slate-900">{scoreDisplay}</span>
          <span className="ml-2 text-xs text-slate-500">({scoreLabel})</span>
        </div>
        {cardData.predicts_year && (
          <span className="text-[11px] text-slate-400 font-mono">
            Forecast: {cardData.predicts_year}
          </span>
        )}
      </div>

      {/* Calibration note for Heatwave / Flood */}
      {cardData.note && (
        <div className="mt-2.5 p-2 rounded bg-amber-50/60 border border-amber-200/60 text-[11px] text-amber-800 leading-snug">
          <span className="font-semibold">Note: </span>
          {cardData.note}
        </div>
      )}

      {/* Top factors breakdown */}
      {cardData.top_factors && cardData.top_factors.length > 0 && (
        <div className="mt-3">
          <span className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider">
            Key Contributing Factors:
          </span>
          <div className="mt-1.5 space-y-1">
            {cardData.top_factors.slice(0, 3).map((f, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between text-xs bg-slate-50 px-2 py-1 rounded border border-slate-100"
              >
                <span className="font-mono text-slate-600 truncate max-w-[180px]">
                  {f.feature}
                </span>
                <span className="font-semibold text-slate-800">
                  {typeof f.value === 'number' ? f.value.toFixed(2) : String(f.value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Collapsible Actionable Guidance */}
      <div className="mt-3 pt-2 border-t border-slate-100">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center justify-between w-full text-xs font-medium text-slate-600 hover:text-slate-900"
        >
          <span>Traveler Recommendations ({config.advice.length})</span>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {expanded && (
          <ul className="mt-2 space-y-1.5 text-xs text-slate-600">
            {config.advice.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 mt-0.5 shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
