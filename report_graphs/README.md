# Report Graphs for 30-Page Forest Fire Risk Detection Report

This folder contains all graphs generated for the report.

## Generated PNGs (Python)

| File | Graph # | Description |
|------|---------|-------------|
| graph1_fires_by_continent.png | 1 | Bar chart - fires by continent |
| graph4_detection_methods.png | 4 | Heatmap - detection methods comparison |
| graph6_feature_matrix.png | 6 | Heatmap - feature comparison (our system vs tools) |
| graph8_fire_density.png | 8 | Fire density by lat/lon |
| graph9_daily_fires.png | 9 | Line chart - daily fire count |
| graph10_brightness_hist.png | 10 | Histogram - brightness distribution |
| graph11_risk_pie.png | 11 | Pie chart - High/Medium/Low risk |
| graph12_frp_brightness_scatter.png | 12 | Scatter - FRP vs brightness |
| graph15_vector_direction.png | 15 | Vector direction - fire spread |
| graph17_feature_importance.png | 17 | Bar chart - Random Forest feature importance |
| graph21_cache_performance.png | 21 | Bar - cache before/after |
| graph22_fire_status.png | 22 | Bar - Growing/Stable/Diminishing |
| graph23_confidence_hist.png | 23 | Histogram - confidence distribution |
| graph26_response_times.png | 26 | Bar - response times by endpoint |
| graph27_metrics_dashboard.png | 27 | 4-panel metrics dashboard |

## Mermaid Diagrams (diagrams.md)

Copy the code blocks into [mermaid.live](https://mermaid.live) or draw.io:
- **Graph 2:** System Architecture
- **Graph 3:** MODIS Satellite Diagram
- **Graph 7:** Data Schema (ER)
- **Graph 13:** Classification Flowchart
- **Graph 14:** Fire Tracking Flowchart
- **Graph 15:** (Also generated as PNG)
- **Graph 16:** ML Pipeline
- **Graph 18:** API Endpoints
- **Graph 19:** Component Hierarchy
- **Graph 20:** Data Flow Diagram

## Manual Screenshots (Graphs 24 & 25)

- **Graph 24:** Predicted Risk Zones Map — Run the app, open the Future Forecast view, take a screenshot
- **Graph 25:** Live vs Historical Comparison — Two side-by-side maps: historical (Jan 13, 2025) and live data

## Regenerating Graphs

```bash
python generate_report_graphs.py
```

## API for Report Data

The `/api/report-stats` endpoint returns JSON with `daily_fires`, `risk_counts`, `brightness_bins`, `fire_tracking`, `feature_importance`, `performance`, etc. for custom chart generation (e.g., with Chart.js).
