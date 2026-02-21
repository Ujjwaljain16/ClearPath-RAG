/**
 * Centralized platform configuration.
 * Change URL at one place and the rest of the application adapts.
 */

// Reads the base URL from the environment if provided (e.g., in production)
// Default to an empty string for relative paths in standard deployment.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string) || "";

/**
 * Technical Specs for Clearpath Nexus
 */
export const PLATFORM_IDENTITY = {
    name: "Clearpath Nexus",
    version: "1.0.0",
    engine: "Grounded Hybrid RAG"
};
