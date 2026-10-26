'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Search, MapPin, Calendar, Loader2, Sparkles } from 'lucide-react';
import { searchDestinations } from '../lib/api';
import { POPULAR_PRESETS } from '../lib/constants';

export default function BottomSearchBar({
  selectedDistrict,
  onSelectDistrict,
  travelDate,
  onDateChange,
  onSearch,
  loading,
}) {
  const [placeQuery, setPlaceQuery] = useState(
    selectedDistrict?.district_name
      ? `${selectedDistrict.district_name}, ${selectedDistrict.state_name}`
      : 'Agra, Uttar Pradesh'
  );
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Sync place query if selectedDistrict changes externally (e.g. clicked on map)
  useEffect(() => {
    if (selectedDistrict?.district_name) {
      setPlaceQuery(
        `${selectedDistrict.district_name}, ${selectedDistrict.state_name || ''}`
      );
    }
  }, [selectedDistrict]);

  // Handle outside click for autocomplete dropdown
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Debounced search against actual backend autocomplete endpoint
  useEffect(() => {
    if (!placeQuery || placeQuery.trim().length < 2) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const res = await searchDestinations(placeQuery, 8);
        setResults(res?.results || []);
        if (res?.results && res.results.length > 0) {
          setIsOpen(true);
        }
      } catch (err) {
        console.warn('Search autocomplete warning:', err);
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [placeQuery]);

  const handleSelect = (item) => {
    onSelectDistrict(item);
    setPlaceQuery(`${item.district_name}, ${item.state_name}`);
    setIsOpen(false);
  };

  const handlePresetSelect = (preset) => {
    const item = {
      census_district_code: preset.code,
      code: preset.code,
      district_name: preset.name,
      state_name: preset.state,
      latitude: preset.lat,
      longitude: preset.lon,
    };
    onSelectDistrict(item);
    setPlaceQuery(`${preset.name}, ${preset.state}`);
    setIsOpen(false);
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-2">
      {/* Quick Select Preset Pills */}
      <div className="flex items-center justify-center gap-1.5 overflow-x-auto pb-1 text-xs">
        <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider hidden sm:inline mr-1">
          Quick Hubs:
        </span>
        {POPULAR_PRESETS.slice(0, 6).map((preset) => (
          <button
            key={preset.code}
            onClick={() => handlePresetSelect(preset)}
            className="px-2.5 py-0.5 rounded-full bg-white hover:bg-slate-200 border border-slate-300 text-slate-700 text-[11px] font-medium whitespace-nowrap transition-all shadow-sm"
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Main Bottom Search Bar directly adhering to wireframe image */}
      <div
        ref={dropdownRef}
        className="relative bg-navy-900 rounded-xl p-3 sm:p-4 shadow-xl border border-navy-800"
      >
        <div className="flex flex-col sm:flex-row items-center gap-3">
          {/* Input 1: Enter Place with live autocomplete */}
          <div className="relative w-full sm:flex-1">
            <div className="relative">
              <input
                type="text"
                value={placeQuery}
                onChange={(e) => {
                  setPlaceQuery(e.target.value);
                  setIsOpen(true);
                }}
                onFocus={() => {
                  if (results.length > 0) setIsOpen(true);
                }}
                placeholder="Enter Place"
                className="w-full h-11 px-3.5 pl-9 rounded bg-white text-slate-900 placeholder:text-slate-500 text-sm font-medium border border-slate-300 focus:outline-none focus:ring-2 focus:ring-skybtn-400"
              />
              <MapPin className="absolute left-3 top-3 w-4 h-4 text-slate-400" />
              {isSearching && (
                <Loader2 className="absolute right-3 top-3 w-4 h-4 text-slate-400 animate-spin" />
              )}
            </div>

            {/* Dropdown Results */}
            {isOpen && results.length > 0 && (
              <div className="absolute bottom-full mb-2 left-0 w-full bg-white rounded-lg border border-slate-300 shadow-2xl max-h-60 overflow-y-auto z-50">
                <div className="p-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-100">
                  Matching Indian Districts ({results.length})
                </div>
                {results.map((r) => (
                  <button
                    key={r.id || r.census_district_code}
                    onClick={() => handleSelect(r)}
                    className="w-full text-left px-3 py-2 hover:bg-sky-50 text-xs flex items-center justify-between border-b border-slate-50 last:border-0"
                  >
                    <div>
                      <span className="font-semibold text-slate-900">
                        {r.district_name}
                      </span>
                      <span className="text-slate-500 ml-1.5">
                        ({r.state_name})
                      </span>
                    </div>
                    {r.elevation_m && (
                      <span className="text-[10px] text-slate-400 font-mono">
                        {r.elevation_m}m
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Input 2: DD/mm/yyyy Date Input */}
          <div className="relative w-full sm:w-48">
            <input
              type="date"
              value={travelDate}
              onChange={(e) => onDateChange(e.target.value)}
              className="w-full h-11 px-3 pl-9 rounded bg-white text-slate-900 text-sm font-medium border border-slate-300 focus:outline-none focus:ring-2 focus:ring-skybtn-400"
              placeholder="DD/mm/yyyy"
            />
            <Calendar className="absolute left-3 top-3 w-4 h-4 text-slate-400" />
          </div>

          {/* Button: Bright Cyan/Blue Search Button matching wireframe */}
          <button
            onClick={onSearch}
            disabled={loading}
            className="w-full sm:w-36 h-11 rounded bg-skybtn-500 hover:bg-skybtn-400 active:bg-skybtn-600 text-navy-950 font-extrabold text-base tracking-wide flex items-center justify-center gap-2 transition-all shadow-md disabled:opacity-70 cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Searching</span>
              </>
            ) : (
              <span>Search</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
