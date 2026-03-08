# Security & Privacy

## Privacy
Smart Surveillance Sorter runs entirely locally. It does not send or share any data externally. No analytics, no telemetry, no cloud services.

External connections are made only for:
- Downloading YOLO models from HuggingFace/Ultralytics (first run only, cached locally)
- Downloading CLIP/BLIP models from HuggingFace (first run only, cached locally)
- Geocoding city name to coordinates via Nominatim/OpenStreetMap (once at first startup or at city change, only the city name is sent)
- Connecting to local Ollama instance (127.0.0.1 only, never external)

## Vulnerabilities
To report a security vulnerability, open a GitHub issue or contact the maintainer directly.
