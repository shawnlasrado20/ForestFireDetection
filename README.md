# 🔥 Forest Fire Risk Detection System

A full-stack web application that visualizes and analyzes forest fire risk using real NASA FIRMS (Fire Information for Resource Management System) MODIS satellite data.

## 🌟 Features

- **Real-time Data Analysis**: Processes NASA FIRMS MODIS satellite data
- **Risk Classification**: Automatically classifies fire risk based on brightness temperature
  - 🔴 **High Risk**: Brightness > 330K
  - 🟠 **Medium Risk**: Brightness 300-330K
  - 🟢 **Low Risk**: Brightness < 300K
- **Interactive Map**: Leaflet.js-powered visualization with color-coded markers
- **Modern Dashboard**: Professional UI with real-time statistics
- **Responsive Design**: Works on desktop and mobile devices
- **REST API**: Clean backend API for data processing

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, or Edge)

## 🚀 Installation & Setup

### 1. Clone or Navigate to the Project Directory

```bash
cd /Users/shawnlasrado/Developer/forestfiredetection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install Flask==3.0.0 pandas==2.1.4
```

### 3. Verify Data File

Ensure the CSV file `fire_nrt_M-C61_565334.csv` is in the project root directory.

### 4. Run the Application

```bash
python app.py
```

You should see output like:

```
CSV file found: fire_nrt_M-C61_565334.csv
Starting Forest Fire Risk Detection System...
 * Running on http://0.0.0.0:5001
```

### 5. Open in Browser

Navigate to: **http://127.0.0.1:5001**

> **Note**: If you see "Port already in use" error, the app will automatically try port 5001 instead of 5000 (common on macOS due to AirPlay Receiver).

## 📁 Project Structure

```
forestfiredetection/
│
├── app.py                          # Flask backend application
├── fire_nrt_M-C61_565334.csv      # NASA FIRMS MODIS data
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/
│   └── index.html                  # Main dashboard HTML
│
└── static/
    ├── css/
    │   └── style.css              # Styling and theme
    └── js/
        └── app.js                 # Frontend JavaScript logic
```

## 🎯 How to Use

1. **Launch the application** by running `python app.py`
2. **Open your browser** to http://127.0.0.1:5001
3. **Click "Analyze Fire Risk"** button in the sidebar
4. **View the results**:
   - Color-coded markers appear on the map
   - Statistics show risk distribution
   - Click any marker to see detailed information

## 🔌 API Endpoints

### `GET /analyze`

Returns fire risk analysis data in JSON format.

**Response Example:**

```json
{
  "success": true,
  "total_points": 5000,
  "statistics": {
    "high_risk": 245,
    "medium_risk": 1832,
    "low_risk": 2923
  },
  "data": [
    {
      "latitude": 32.23001,
      "longitude": 51.4333,
      "brightness": 311.05,
      "risk": "Medium"
    }
    // ... more data points
  ]
}
```

### `GET /health`

Health check endpoint to verify system status.

## ⚙️ Configuration

You can modify the following settings in `app.py`:

- **MAX_ROWS**: Number of data points to load (default: 5000)
  ```python
  MAX_ROWS = 5000  # Increase for more data points
  ```

- **Port**: Change the server port (default: 5001)
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)
  ```

## 🎨 Map Themes

The application uses a dark theme map from CartoDB. To change the map style, edit `static/js/app.js`:

```javascript
// Current: Dark theme
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {...})

// Alternative: Light theme
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {...})

// Alternative: Satellite view
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {...})
```

## 📊 Data Source

This application uses NASA FIRMS (Fire Information for Resource Management System) data:

- **Satellite**: MODIS (Moderate Resolution Imaging Spectroradiometer)
- **Data Type**: Near Real-Time (NRT) Fire Detection
- **Brightness**: Surface temperature in Kelvin
- **Source**: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)

## 🛠️ Troubleshooting

### Issue: "CSV file not found"

**Solution**: Ensure `fire_nrt_M-C61_565334.csv` is in the same directory as `app.py`

### Issue: "Module not found" errors

**Solution**: Install required packages:

```bash
pip install Flask pandas
```

### Issue: Port 5001 already in use

**Solution**: Change the port in `app.py` or kill the process using port 5001:

```bash
# Find process using port 5001
lsof -i :5001

# Kill the process (replace PID with actual process ID)
kill -9 PID
```

Or disable AirPlay Receiver on macOS:
System Settings → General → AirDrop & Handoff → Disable "AirPlay Receiver"

### Issue: Map not loading

**Solution**: Check your internet connection (Leaflet.js tiles require internet access)

## 🎓 Academic Use

This prototype is designed for academic demonstration and research purposes. It showcases:

- Full-stack web development (Flask + HTML/CSS/JavaScript)
- Real-world data processing and visualization
- REST API design and implementation
- Interactive mapping with Leaflet.js
- Risk assessment algorithms

## 📝 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- NASA FIRMS for providing satellite fire detection data
- Leaflet.js for map visualization
- Flask framework for backend
- CartoDB for map tiles

---

**Built with ❤️ for Forest Fire Detection and Prevention**
