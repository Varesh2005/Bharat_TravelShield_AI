/**
 * Bharat-TravelShield AI — Constants & UI Configuration
 * Aligned with backend data contracts & scientific governance rules
 */

export const HAZARD_CONFIG = {
  malaria: {
    id: 'malaria',
    name: 'Malaria Risk',
    badgeColor: 'border-emerald-500/40 bg-emerald-50 text-emerald-700',
    description: 'Vector-borne Anopheles transmission risk based on post-monsoon surface pooling & historical surveillance.',
    targetLabel: 'High reported incidence next calendar year',
    advice: [
      'Carry DEET / Picaridin mosquito repellent (20-30% concentration).',
      'Wear permethrin-treated long sleeves and trousers during dusk and dawn.',
      'Sleep under insecticide-treated bed nets (LLINs) in non-air-conditioned accommodations.',
      'Consult travel clinic regarding chemoprophylaxis (Atovaquone-proguanil or Doxycycline) if traveling during peak monsoon.'
    ]
  },
  heatwave: {
    id: 'heatwave',
    name: 'Heatwave Emergency',
    badgeColor: 'border-amber-500/40 bg-amber-50 text-amber-700',
    description: 'Meteorological extreme thermal event risk based on IMD climatological threshold exceedances.',
    targetLabel: 'Extreme heatwave emergency days (summer cycle)',
    advice: [
      'Avoid strenuous outdoor activities between 11:00 AM and 4:00 PM.',
      'Carry oral rehydration salts (ORS), electrolyte fluids, and minimum 3-4L water daily.',
      'Wear lightweight, loose-fitting, light-colored cotton garments and wide-brimmed hats.',
      'Watch for symptoms of heat exhaustion (dizziness, nausea, profuse sweating) and seek immediate shade/cooling.'
    ]
  },
  flood: {
    id: 'flood',
    name: 'Hydrological Flood',
    badgeColor: 'border-blue-500/40 bg-blue-50 text-blue-700',
    description: 'Hydrological inundation surge based on cumulative rainfall and low-lying topography.',
    targetLabel: 'Extreme inundation and localized flooding surge',
    advice: [
      'Check local NDMA / IMD district red/orange alerts before setting out.',
      'Avoid riverbank tourism, water crossings, and low-lying waterlogged arterial roads.',
      'Keep waterproof pouches for documents, emergency medication, and power banks.',
      'Maintain contact with hotel staff or local disaster management cell (Helpline: 1070/1077).'
    ]
  },
  dengue: {
    id: 'dengue',
    name: 'Dengue Advisory',
    badgeColor: 'border-rose-500/40 bg-rose-50 text-rose-700',
    description: 'State-level epidemiological surveillance alert (district ANN withheld due to scientific data governance).',
    targetLabel: 'State-level epidemiological surge advisory',
    advice: [
      'Aedes mosquitoes bite actively during daytime (sunrise to sunset); apply repellent during outdoor daytime sightseeing.',
      'Avoid standing near artificial water containers, flower pots, and unsealed water tanks.',
      'Seek prompt medical evaluation if experiencing high fever with retro-orbital eye pain and joint aches.'
    ]
  }
};

export const POPULAR_PRESETS = [
  { code: 146, name: 'Agra', state: 'Uttar Pradesh', desc: 'Taj Mahal Hub', lat: 27.1767, lon: 78.0081 },
  { code: 90, name: 'Central Delhi', state: 'Delhi', desc: 'National Capital', lat: 28.6139, lon: 77.2090 },
  { code: 518, name: 'Mumbai', state: 'Maharashtra', desc: 'Coastal Metropolis', lat: 18.9388, lon: 72.8354 },
  { code: 572, name: 'Bengaluru Urban', state: 'Karnataka', desc: 'Silicon Plateau', lat: 12.9716, lon: 77.5946 },
  { code: 585, name: 'North Goa', state: 'Goa', desc: 'Monsoon Beach Hub', lat: 15.4989, lon: 73.8278 },
  { code: 194, name: 'Varanasi', state: 'Uttar Pradesh', desc: 'Ghats & Heritage', lat: 25.3176, lon: 82.9739 },
  { code: 109, name: 'Jaipur', state: 'Rajasthan', desc: 'Pink City', lat: 26.9124, lon: 75.7873 },
  { code: 536, name: 'Hyderabad', state: 'Telangana', desc: 'Deccan Hub', lat: 17.3850, lon: 78.4867 },
  { code: 598, name: 'Ernakulam (Kochi)', state: 'Kerala', desc: 'Coastal Port', lat: 9.9816, lon: 76.2999 },
  { code: 25, name: 'Shimla', state: 'Himachal Pradesh', desc: 'Himalayan Ridge', lat: 31.1048, lon: 77.1734 },
];

export const RISK_TIERS = {
  Low: { label: 'Low', badge: 'bg-emerald-100 text-emerald-800 border-emerald-300' },
  Moderate: { label: 'Moderate', badge: 'bg-amber-100 text-amber-800 border-amber-300' },
  Elevated: { label: 'Elevated', badge: 'bg-orange-100 text-orange-800 border-orange-300' },
  High: { label: 'High', badge: 'bg-rose-100 text-rose-800 border-rose-300' },
  Severe: { label: 'Severe', badge: 'bg-red-200 text-red-900 border-red-500' },
};
