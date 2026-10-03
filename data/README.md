# Data layout

This directory is for project data only. It is intentionally separated from application code.

## Recommended structure

- `private/` - raw database exports, local dev datasets, and other production-sensitive files
- `public/` - sanitized or sample data that may be shared externally or used for demos
- `audio/` - generated audio files or audio samples used by the app or future research

## Safety rules

- Do not store raw PostgreSQL dumps in `public/`
- Do not commit personal or sensitive user data to GitHub
- Only put public-safe, anonymized, or sample data in `public/`
- Keep all real app data in `private/` or in your database environment

## Notes

This folder is a future-facing structure for data governance. It is not a requirement to upload data to GitHub right now.
