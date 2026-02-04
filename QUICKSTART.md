# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

You should see:
```
CSV file found: fire_nrt_M-C61_565334.csv
Starting Forest Fire Risk Detection System...
Access the dashboard at: http://127.0.0.1:5001
```

### Step 3: Open Your Browser

Go to: **http://127.0.0.1:5001**

---

## Using the Dashboard

1. **Click "Analyze Fire Risk"** button in the sidebar
2. Wait a few seconds while the system processes 5,000+ data points
3. **Explore the map**:
   - 🔴 Red dots = High Risk (>330K)
   - 🟠 Orange dots = Medium Risk (300-330K)
   - 🟢 Green dots = Low Risk (<300K)
4. **Click any marker** to see detailed information
5. **View statistics** in the sidebar

---

## What You're Seeing

- **Real NASA Satellite Data**: MODIS fire detection from space
- **Brightness Temperature**: Measured in Kelvin from satellite sensors
- **Risk Classification**: Automated AI-based risk assessment
- **Global Coverage**: Fire detections from around the world

---

## Tips

- Zoom in/out on the map to explore specific regions
- Use the legend to understand risk levels
- Check the statistics panel for data overview
- Refresh the page to reload data

---

## Troubleshooting

**Problem**: Port already in use  
**Solution**: The app uses port 5001 to avoid conflicts

**Problem**: No markers showing  
**Solution**: Click "Analyze Fire Risk" button first

**Problem**: Map not loading  
**Solution**: Check internet connection (map tiles require internet)

---

**Ready to explore! 🌍🔥**
