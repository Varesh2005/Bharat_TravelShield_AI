/**
 * Bharat-TravelShield AI — API Client Service
 * Author: Rosemariya (Frontend / UI)
 * Integrates FastAPI REST microservice (Rajdeep) & Firestore persistence (Pranav)
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });

    if (!res.ok) {
      let errData = {};
      try {
        errData = await res.json();
      } catch (e) {
        errData = { detail: res.statusText };
      }
      const error = new Error(errData.detail || errData.message || `HTTP error ${res.status}`);
      error.status = res.status;
      error.data = errData;
      throw error;
    }

    return await res.json();
  } catch (err) {
    console.warn(`[API] Network or parsing notice for ${endpoint}:`, err.message);
    throw err;
  }
}

// -----------------------------------------------------------------------------
// System & Health
// -----------------------------------------------------------------------------

export async function getSystemHealth() {
  return fetchJson('/api/v1/health');
}

export async function getSystemOverview() {
  return fetchJson('/');
}

// -----------------------------------------------------------------------------
// Destinations & Search
// -----------------------------------------------------------------------------

export async function searchDestinations(query, limit = 10) {
  if (!query || !query.trim()) return { query: '', count: 0, results: [] };
  return fetchJson(`/api/v1/destinations/search?q=${encodeURIComponent(query.trim())}&limit=${limit}`);
}

export async function listDestinations(limit = 50) {
  return fetchJson(`/api/v1/destinations?limit=${limit}`);
}

export async function getDestinationById(id) {
  return fetchJson(`/api/v1/destinations/${encodeURIComponent(id)}`);
}

export async function getDestinationSummary(districtCode, options = {}) {
  const { startDate, endDate, includeAnn = false } = options;
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  if (includeAnn) params.append('include_ann', 'true');
  const qs = params.toString() ? `?${params.toString()}` : '';
  return fetchJson(`/api/v1/destinations/district/${districtCode}/summary${qs}`);
}

// -----------------------------------------------------------------------------
// Risk Intelligence & Predictions
// -----------------------------------------------------------------------------

export async function predictMultiHazard(districtCode, userId = 'traveler_web', featureYear = null) {
  return fetchJson('/api/v1/predict/multi-hazard', {
    method: 'POST',
    body: JSON.stringify({
      district_code: Number(districtCode),
      user_id: userId,
      feature_year: featureYear ? Number(featureYear) : null,
    }),
  });
}

export async function getSupportedHazards() {
  return fetchJson('/api/v1/hazards');
}

// -----------------------------------------------------------------------------
// Counterfactual Simulator & Maps
// -----------------------------------------------------------------------------

export async function simulateScenario(hazard, districtCode, deltas, baselineYear = null) {
  return fetchJson('/api/v1/risk/scenario', {
    method: 'POST',
    body: JSON.stringify({
      hazard: hazard.toLowerCase(),
      district_code: Number(districtCode),
      deltas,
      baseline_year: baselineYear ? Number(baselineYear) : null,
    }),
  });
}

export async function getMapLayer(hazardType) {
  return fetchJson(`/api/v1/map/layers/${encodeURIComponent(hazardType)}`);
}

export async function getResearchLeaderboard() {
  return fetchJson('/api/v1/research/leaderboard');
}

// -----------------------------------------------------------------------------
// History & Auditing
// -----------------------------------------------------------------------------

export async function getWeatherHistory(locationId, limit = 30) {
  return fetchJson(`/api/v1/history/weather/${encodeURIComponent(locationId)}?limit=${limit}`);
}

export async function getDiseaseHistory(locationId, limit = 30) {
  return fetchJson(`/api/v1/history/disease/${encodeURIComponent(locationId)}?limit=${limit}`);
}

export async function getRecentPredictions(limit = 20) {
  return fetchJson(`/api/v1/predictions/recent?limit=${limit}`);
}
