# Report Diagrams - Mermaid Code

Copy these into draw.io, Mermaid Live Editor (mermaid.live), or use in Markdown.

---

## Graph 3: MODIS Satellite Diagram

```mermaid
flowchart TB
    subgraph Earth[Earth]
        Surface[Surface]
    end

    subgraph MODIS[MODIS Satellite]
        Orbit[Orbit 705km]
        Sensor[Optical Sensor]
        Band[36 Spectral Bands]
        Orbit --> Sensor
        Sensor --> Band
    end

    Surface -->|Radiance| Sensor
    Band -->|Thermal IR| FIRMS[FIRMS Processing]
```

---

## Graph 7: Data Schema (ER Diagram)

```mermaid
erDiagram
    FIRM_DETECTION {
        string acq_date
        string acq_time
        float latitude
        float longitude
        float brightness
        string satellite
        string instrument
        string daynight
    }
    FIRM_DETECTION ||--o| FRP : has
    FRP {
        float frp_MW
        int confidence
    }
    FIRM_DETECTION ||--o| SCAN : has
    SCAN {
        float scan
        float track
    }
```

---

## Graph 2: System Architecture

```mermaid
flowchart LR
    User[User Browser]
    Flask[Flask API]
    Pandas[Pandas]
    ML[scikit-learn]
    NASA[NASA FIRMS API]
    CSV[(CSV Database)]

    User -->|HTTP| Flask
    Flask --> Pandas
    Flask --> ML
    Flask -->|Fetch| NASA
    Pandas --> CSV
    Flask -->|JSON| User
```

---

## Graph 13: Risk Classification Flowchart

```mermaid
flowchart TD
    A[Input: Brightness K] --> B{Brightness > 330?}
    B -->|Yes| C[High Risk]
    B -->|No| D{Brightness >= 300?}
    D -->|Yes| E[Medium Risk]
    D -->|No| F[Low Risk]
    C --> G[Output]
    E --> G
    F --> G
```

---

## Graph 14: Fire Tracking Flowchart

```mermaid
flowchart TD
    A[Load Fire Data] --> B[Create Grid Cells 0.1 deg]
    B --> C[For Each Date]
    C --> D[Map Fires to Grid Cells]
    D --> E[Track Cell History]
    E --> F{Cell has fires on 2+ days?}
    F -->|Yes| G[Mark Persistent Fire]
    F -->|No| H[Skip]
    G --> I[Calculate Status: Growing/Stable/Diminishing]
    I --> J[Output Fire Tracks]
```

---

## Graph 16: ML Pipeline

```mermaid
flowchart LR
    CSV[Historical CSV] --> FE[Feature Engineering]
    FE --> Grid[Grid Cells]
    FE --> Past7[Past 7 Days Stats]
    Grid --> Train[Train Random Forest]
    Past7 --> Train
    Train --> Model[Model]
    Recent[Recent Data] --> Predict[Predict]
    Model --> Predict
    Predict --> Zones[Risk Zones]
    Zones --> Map[Display on Map]
```

---

## Graph 18: API Endpoints

```mermaid
flowchart TD
    Client[Client]
    Analyze[/analyze]
    Live[/analyze/live]
    Track[/predict/tracking]
    Spread[/predict/spread]
    Forecast[/predict/future-forecast]
    Health[/health]
    Report[/api/report-stats]

    Client --> Analyze
    Client --> Live
    Client --> Track
    Client --> Spread
    Client --> Forecast
    Client --> Health
    Client --> Report

    Analyze --> CSV[(CSV)]
    Live --> NASA[NASA API]
    Track --> CSV
    Spread --> Track
    Forecast --> ML[ML Model]
```

---

## Graph 19: Component Hierarchy

```mermaid
flowchart TD
    App[Forest Fire App]
    Sidebar[Sidebar]
    Map[Map Area]

    App --> Sidebar
    App --> Map

    Sidebar --> Header[Header]
    Sidebar --> AnalyzeBtn[Analyze Button]
    Sidebar --> Toggle[Live/Historical Toggle]
    Sidebar --> Timeline[Timeline Controls]
    Sidebar --> Predict[Prediction Panel]
    Sidebar --> Legend[Legend]
    Sidebar --> Stats[Statistics]

    Map --> Leaflet[Leaflet Map]
    Map --> Markers[Fire Markers]
    Map --> PredictLayer[Prediction Layer]
```

---

## Graph 20: Data Flow

```mermaid
flowchart LR
    subgraph Sources
        CSV[CSV File]
        NASA[NASA API]
    end

    subgraph Backend
        Load[Load Data]
        Process[Process]
        Classify[Classify Risk]
        Track[Track Fires]
        ML[ML Predict]
        JSON[JSON Response]
    end

    subgraph Frontend
        Fetch[Fetch API]
        Parse[Parse JSON]
        Plot[Plot on Map]
    end

    CSV --> Load
    NASA --> Load
    Load --> Process
    Process --> Classify
    Process --> Track
    Process --> ML
    Classify --> JSON
    Track --> JSON
    ML --> JSON
    JSON --> Fetch
    Fetch --> Parse
    Parse --> Plot
```
