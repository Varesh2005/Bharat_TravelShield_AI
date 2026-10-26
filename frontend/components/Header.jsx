'use client';

import React from 'react';
import { ShieldCheck, Activity } from 'lucide-react';

export default function Header({ backendStatus }) {
  return (
    <header className="w-full bg-navy-900 text-white shadow-md border-b border-navy-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Title strictly matching the wireframe */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-skybtn-500/20 border border-skybtn-500/40 flex items-center justify-center text-skybtn-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
            Bharat_Travelshield_Ai
          </h1>
        </div>

        {/* Live System Indicator */}
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              backendStatus?.online ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
          />
          <span className="hidden sm:inline font-mono">
            {backendStatus?.online ? backendStatus.dbMode || 'Live' : 'Backend Initializing'}
          </span>
        </div>
      </div>
    </header>
  );
}
