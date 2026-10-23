// Bharat-TravelShield AI — Firebase Web App Client Configuration
// Configured for Rosemariya (Frontend) & Pranav (Integration)

import { initializeApp, getApps, getApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";

export const firebaseConfig = {
  apiKey: "AIzaSyDNxjE4ZOs8Lgp3bdca1oC21t0tn2J9nlY",
  authDomain: "bharat-travelshield-ai.firebaseapp.com",
  projectId: "bharat-travelshield-ai",
  storageBucket: "bharat-travelshield-ai.firebasestorage.app",
  messagingSenderId: "985133965440",
  appId: "1:985133965440:web:9d5817c737a1d0ab360665",
  measurementId: "G-CCL2XSKW6Y"
};

// Initialize Firebase for SSR / Client safely
export const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApp();
export const db = getFirestore(app);
export const auth = getAuth(app);

export default app;
