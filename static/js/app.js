// Forest Fire Risk Detection System - Frontend JavaScript

// Global variables
let map;
let markersLayer;
let fireData = [];
let fireDataByDate = {}; // Fires grouped by date
let allMarkers = {}; // All markers by date for reuse

// Prediction layers
let spreadLayer;
let riskHeatmapLayer;
let trackingData = null;
let spreadData = null;
let forecastData = null;

// Prediction state
let predictionState = {
    showSpread: false,
    showForecast: false,
    showTracking: false
};

// Timeline state
let timelineData = {
    dates: [],
    currentDateIndex: 0,
    isPlaying: false,
    playbackSpeed: 500,
    intervalId: null
};

// Color mapping for risk levels
const RISK_COLORS = {
    'High': '#ef4444',
    'Medium': '#f97316',
    'Low': '#22c55e'
};

// Fire lifespan colors
const LIFESPAN_COLORS = {
    'new': '#ecf95a',      // Bright yellow
    'persistent': '#f97316', // Orange
    'longterm': '#ef4444'   // Red
};

// Initialize the map
function initMap() {
    // Create map centered on a global view
    map = L.map('map').setView([20, 0], 2);
    
    // Add tile layer (dark theme)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors © CARTO',
        maxZoom: 19
    }).addTo(map);
    
    // Create a feature group for markers (supports getBounds)
    markersLayer = L.featureGroup().addTo(map);
    
    // Create prediction layers
    spreadLayer = L.layerGroup().addTo(map);
    riskHeatmapLayer = L.layerGroup().addTo(map);
    
    console.log('Map initialized');
}

// Fetch and analyze fire data from backend
async function analyzeFireRisk() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const statistics = document.getElementById('statistics');
    const mapOverlay = document.getElementById('mapOverlay');
    
    try {
        // Show loading state
        analyzeBtn.disabled = true;
        loading.classList.remove('hidden');
        statistics.classList.add('hidden');
        
        // Fetch data from backend
        const response = await fetch('/analyze');
        
        if (!response.ok) {
            throw new Error('Failed to fetch data from server');
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Unknown error occurred');
        }
        
        // Store fire data
        fireData = result.data;
        
        // Group data by date
        groupDataByDate(fireData);
        
        // Initialize timeline
        if (result.timeline && result.timeline.dates) {
            timelineData.dates = result.timeline.dates;
            // Start at MOST RECENT date (last index) instead of first
            timelineData.currentDateIndex = result.timeline.dates.length - 1;
            initializeTimeline(result.timeline);
        }
        
        // Update statistics
        updateStatistics(result);
        
        // Start timeline at MOST RECENT date (show current fire state)
        if (timelineData.dates.length > 0) {
            updateMapForDate(timelineData.dates.length - 1);
        } else {
            // Fallback: Plot all data if no timeline
            plotFireData(fireData);
        }
        
        // Hide overlay
        mapOverlay.classList.add('hidden');
        
        // Show statistics
        statistics.classList.remove('hidden');
        
        console.log(`Loaded ${fireData.length} fire data points across ${timelineData.dates.length} days`);
        
    } catch (error) {
        console.error('Error analyzing fire risk:', error);
        alert('Error: ' + error.message);
    } finally {
        // Hide loading state
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

// Group fire data by date for efficient timeline filtering
function groupDataByDate(data) {
    fireDataByDate = {};
    data.forEach(fire => {
        const date = fire.date;
        if (!fireDataByDate[date]) {
            fireDataByDate[date] = [];
        }
        fireDataByDate[date].push(fire);
    });
    console.log(`Grouped fires into ${Object.keys(fireDataByDate).length} dates`);
}

// Initialize timeline controls
function initializeTimeline(timeline) {
    const timelineControls = document.getElementById('timelineControls');
    const timelineSlider = document.getElementById('timelineSlider');
    const currentDateDisplay = document.getElementById('currentDateDisplay');
    
    if (timelineControls) {
        timelineControls.classList.remove('hidden');
    }
    
    // Set slider range and position at END (most recent)
    if (timelineSlider) {
        timelineSlider.max = timeline.dates.length - 1;
        timelineSlider.value = timeline.dates.length - 1;  // Start at end
    }
    
    // Display MOST RECENT date initially
    if (currentDateDisplay && timeline.dates.length > 0) {
        const lastDate = timeline.dates[timeline.dates.length - 1];
        currentDateDisplay.textContent = formatDate(lastDate);
    }
    
    console.log(`Timeline initialized: ${timeline.start_date} to ${timeline.end_date} (starting at most recent)`);
}

// Format date for display (YYYY-MM-DD to readable format)
function formatDate(dateString) {
    const date = new Date(dateString + 'T00:00:00');
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

// Update statistics display
function updateStatistics(result) {
    document.getElementById('totalPoints').textContent = result.total_points.toLocaleString();
    document.getElementById('highRisk').textContent = result.statistics.high_risk.toLocaleString();
    document.getElementById('mediumRisk').textContent = result.statistics.medium_risk.toLocaleString();
    document.getElementById('lowRisk').textContent = result.statistics.low_risk.toLocaleString();
}

// Plot fire data on the map
function plotFireData(data) {
    // Clear existing markers
    markersLayer.clearLayers();
    
    if (data.length === 0) {
        alert('No fire data available to display');
        return;
    }
    
    // Create markers for each data point
    data.forEach(point => {
        const color = RISK_COLORS[point.risk] || '#64748b';
        
        // Create circle marker
        const marker = L.circleMarker(
            [point.latitude, point.longitude],
            {
                radius: 6,
                fillColor: color,
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }
        );
        
        // Create popup content
        const popupContent = createPopupContent(point);
        marker.bindPopup(popupContent);
        
        // Add marker to layer
        marker.addTo(markersLayer);
    });
    
    // Fit map bounds to show all markers
    if (markersLayer.getLayers().length > 0) {
        const bounds = markersLayer.getBounds();
        map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    console.log(`Plotted ${data.length} markers on map`);
}

// Create popup content for markers
function createPopupContent(point) {
    const riskClass = point.risk.toLowerCase();
    
    return `
        <div class="popup-content">
            <h4>Fire Detection Point</h4>
            <p><strong>Risk Level:</strong> <span style="color: ${RISK_COLORS[point.risk]}">${point.risk}</span></p>
            <p><strong>Brightness:</strong> ${point.brightness.toFixed(2)}K</p>
            <p><strong>Latitude:</strong> ${point.latitude.toFixed(4)}°</p>
            <p><strong>Longitude:</strong> ${point.longitude.toFixed(4)}°</p>
        </div>
    `;
}

// ============= TIMELINE ANIMATION FUNCTIONS =============

// Play timeline animation
function playTimeline() {
    console.log('Play timeline called', timelineData);
    if (timelineData.isPlaying) {
        console.log('Already playing, returning');
        return;
    }
    
    if (timelineData.dates.length === 0) {
        console.error('No timeline dates available');
        return;
    }
    
    timelineData.isPlaying = true;
    const playPauseBtn = document.getElementById('playPauseBtn');
    if (playPauseBtn) {
        playPauseBtn.innerHTML = '⏸ Pause';
    }
    
    console.log(`Starting timeline animation from index ${timelineData.currentDateIndex} at ${timelineData.playbackSpeed}ms intervals`);
    
    timelineData.intervalId = setInterval(() => {
        if (timelineData.currentDateIndex < timelineData.dates.length - 1) {
            timelineData.currentDateIndex++;
            console.log(`Advancing to date index ${timelineData.currentDateIndex}`);
            updateMapForDate(timelineData.currentDateIndex);
        } else {
            // Reached the end, stop playing
            console.log('Reached end of timeline');
            pauseTimeline();
        }
    }, timelineData.playbackSpeed);
}

// Pause timeline animation
function pauseTimeline() {
    timelineData.isPlaying = false;
    const playPauseBtn = document.getElementById('playPauseBtn');
    if (playPauseBtn) {
        playPauseBtn.innerHTML = '▶ Play';
    }
    
    if (timelineData.intervalId) {
        clearInterval(timelineData.intervalId);
        timelineData.intervalId = null;
    }
}

// Toggle play/pause
function togglePlayPause() {
    console.log('Toggle play/pause clicked');
    if (timelineData.isPlaying) {
        console.log('Pausing');
        pauseTimeline();
    } else {
        console.log('Playing');
        // If at the end, restart from beginning
        if (timelineData.currentDateIndex >= timelineData.dates.length - 1) {
            console.log('At end, restarting from beginning');
            timelineData.currentDateIndex = 0;
        }
        playTimeline();
    }
}

// Go to specific date by index
function goToDate(dateIndex) {
    console.log(`Go to date: ${dateIndex}`);
    pauseTimeline();
    timelineData.currentDateIndex = parseInt(dateIndex);
    updateMapForDate(timelineData.currentDateIndex);
}

// Update map to show fires for current date
function updateMapForDate(dateIndex) {
    const currentDate = timelineData.dates[dateIndex];
    const timelineSlider = document.getElementById('timelineSlider');
    const currentDateDisplay = document.getElementById('currentDateDisplay');
    
    // Update slider
    if (timelineSlider) {
        timelineSlider.value = dateIndex;
    }
    
    // Update date display
    if (currentDateDisplay) {
        currentDateDisplay.textContent = formatDate(currentDate);
    }
    
    // Clear existing markers
    markersLayer.clearLayers();
    
    // Show fires up to and including current date (with fade for older ones)
    let currentDayCount = 0;
    let historyCount = 0;
    
    for (let i = 0; i <= dateIndex; i++) {
        const date = timelineData.dates[i];
        const firesForDate = fireDataByDate[date] || [];
        const isCurrentDay = (i === dateIndex);
        const daysAgo = dateIndex - i;
        
        firesForDate.forEach(fire => {
            const color = RISK_COLORS[fire.risk] || '#64748b';
            
            // Calculate opacity based on how old the fire is
            let opacity = isCurrentDay ? 1.0 : Math.max(0.2, 1.0 - (daysAgo * 0.1));
            let fillOpacity = isCurrentDay ? 0.8 : Math.max(0.15, 0.8 - (daysAgo * 0.08));
            
            const marker = L.circleMarker(
                [fire.latitude, fire.longitude],
                {
                    radius: isCurrentDay ? 6 : 4,
                    fillColor: color,
                    color: '#fff',
                    weight: isCurrentDay ? 1 : 0.5,
                    opacity: opacity,
                    fillOpacity: fillOpacity,
                    className: isCurrentDay ? 'current-fire' : 'historical-fire'
                }
            );
            
            // Add popup
            const popupContent = `
                <div class="popup-content">
                    <h4>Fire Detection</h4>
                    <p><strong>Date:</strong> ${formatDate(fire.date)}</p>
                    <p><strong>Risk Level:</strong> <span style="color: ${color}">${fire.risk}</span></p>
                    <p><strong>Brightness:</strong> ${fire.brightness.toFixed(2)}K</p>
                    <p><strong>Location:</strong> ${fire.latitude.toFixed(4)}°, ${fire.longitude.toFixed(4)}°</p>
                </div>
            `;
            marker.bindPopup(popupContent);
            
            marker.addTo(markersLayer);
            
            if (isCurrentDay) currentDayCount++;
            else historyCount++;
        });
    }
    
    // Update statistics for current date
    updateTimelineStatistics(dateIndex);
    
    console.log(`Showing ${currentDayCount} current fires, ${historyCount} historical fires for ${currentDate}`);
}

// Update statistics for current timeline position
function updateTimelineStatistics(dateIndex) {
    let totalFires = 0;
    let highRisk = 0;
    let mediumRisk = 0;
    let lowRisk = 0;
    
    // Count fires up to current date
    for (let i = 0; i <= dateIndex; i++) {
        const date = timelineData.dates[i];
        const fires = fireDataByDate[date] || [];
        
        fires.forEach(fire => {
            totalFires++;
            if (fire.risk === 'High') highRisk++;
            else if (fire.risk === 'Medium') mediumRisk++;
            else if (fire.risk === 'Low') lowRisk++;
        });
    }
    
    // Update display
    document.getElementById('totalPoints').textContent = totalFires.toLocaleString();
    document.getElementById('highRisk').textContent = highRisk.toLocaleString();
    document.getElementById('mediumRisk').textContent = mediumRisk.toLocaleString();
    document.getElementById('lowRisk').textContent = lowRisk.toLocaleString();
}

// Change playback speed
function changeSpeed(speed) {
    timelineData.playbackSpeed = parseInt(speed);
    
    // If playing, restart with new speed
    if (timelineData.isPlaying) {
        pauseTimeline();
        playTimeline();
    }
}

// ============= END TIMELINE FUNCTIONS =============

// ============= PREDICTION FUNCTIONS =============

// Fetch fire tracking data
async function loadFireTracking() {
    try {
        const response = await fetch('/predict/tracking');
        const result = await response.json();
        
        if (result.success) {
            trackingData = result.tracking;
            console.log(`Loaded tracking for ${trackingData.total_tracked} fires`);
            updateTrackingStats();
            return trackingData;
        }
    } catch (error) {
        console.error('Error loading fire tracking:', error);
    }
    return null;
}

// Fetch spread predictions
async function loadSpreadPredictions() {
    try {
        const response = await fetch('/predict/spread');
        const result = await response.json();
        
        if (result.success) {
            spreadData = result.spread;
            console.log(`Loaded ${spreadData.total_predictions} spread predictions`);
            return spreadData;
        }
    } catch (error) {
        console.error('Error loading spread predictions:', error);
    }
    return null;
}

// Fetch risk forecast (REAL FUTURE PREDICTION for Feb 2026)
async function loadRiskForecast() {
    try {
        console.log('Loading FUTURE forecast for Feb 2026...');
        const response = await fetch('/predict/future-forecast');  // Use new endpoint
        const result = await response.json();
        
        if (result.success) {
            forecastData = result.forecast;
            console.log(`Loaded ${forecastData.total_zones} predicted fire zones for ${forecastData.target_date}`);
            console.log(`Based on data from: ${forecastData.based_on_dates}`);
            return forecastData;
        }
    } catch (error) {
        console.error('Error loading future forecast:', error);
    }
    return null;
}

// Update tracking statistics display
function updateTrackingStats() {
    if (!trackingData) return;
    
    const trackedFiresEl = document.getElementById('trackedFires');
    const growingFiresEl = document.getElementById('growingFires');
    const riskZonesEl = document.getElementById('riskZones');
    
    if (trackedFiresEl) trackedFiresEl.textContent = trackingData.total_tracked;
    if (growingFiresEl) growingFiresEl.textContent = trackingData.growing;
    if (riskZonesEl && forecastData) riskZonesEl.textContent = forecastData.high_risk_count || 0;
}

// Visualize fire spread predictions
function visualizeSpread() {
    spreadLayer.clearLayers();
    
    if (!spreadData || !predictionState.showSpread) return;
    
    spreadData.predictions.forEach(pred => {
        const [lat, lon] = pred.current_location;
        const [predLat, predLon] = pred.predicted_location;
        
        // Draw arrow from current to predicted location
        const arrowPoints = [
            [lat, lon],
            [predLat, predLon]
        ];
        
        // Color by confidence
        const color = pred.confidence > 0.6 ? '#22c55e' : pred.confidence > 0.4 ? '#f97316' : '#ef4444';
        
        // Draw line
        const arrow = L.polyline(arrowPoints, {
            color: color,
            weight: 3,
            opacity: 0.8,
            dashArray: '10, 5'
        }).addTo(spreadLayer);
        
        // Add arrowhead using a marker
        const arrowIcon = L.divIcon({
            html: `<div style="color: ${color}; font-size: 24px; transform: rotate(${pred.direction_degrees}deg);">➤</div>`,
            className: 'arrow-icon',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        L.marker([predLat, predLon], { icon: arrowIcon }).addTo(spreadLayer);
        
        // Add prediction circle (uncertainty)
        L.circle([predLat, predLon], {
            radius: pred.uncertainty_radius_km * 1000,
            color: color,
            fillColor: color,
            fillOpacity: 0.1,
            weight: 1
        }).addTo(spreadLayer);
        
        // Popup with prediction info
        const popupContent = `
            <div class="popup-content">
                <h4>Fire Spread Prediction</h4>
                <p><strong>Direction:</strong> ${pred.direction}</p>
                <p><strong>Velocity:</strong> ${pred.velocity_km_day} km/day</p>
                <p><strong>Confidence:</strong> ${(pred.confidence * 100).toFixed(0)}%</p>
            </div>
        `;
        arrow.bindPopup(popupContent);
    });
    
    console.log(`Visualized ${spreadData.predictions.length} spread predictions`);
}

// Visualize risk forecast heatmap (REAL FUTURE PREDICTION)
function visualizeRiskForecast() {
    riskHeatmapLayer.clearLayers();
    
    if (!forecastData || !predictionState.showForecast) return;
    
    forecastData.risk_zones.forEach(zone => {
        const [lat, lon] = zone.center;
        const [[minLat, minLon], [maxLat, maxLon]] = zone.bounds;
        
        // Color by risk level
        const colors = {
            'high': '#ef4444',
            'medium': '#f97316',
            'low': '#fbbf24'
        };
        const color = colors[zone.risk_level];
        
        // Draw risk zone rectangle
        const bounds = [[minLat, minLon], [maxLat, maxLon]];
        const rect = L.rectangle(bounds, {
            color: color,
            fillColor: color,
            fillOpacity: 0.35,
            weight: 2,
            opacity: 0.9,
            dashArray: '5, 5'  // Dashed to show it's a prediction
        }).addTo(riskHeatmapLayer);
        
        // Popup with FUTURE forecast info
        const popupContent = `
            <div class="popup-content">
                <h4>🔮 Future Fire Prediction</h4>
                <p><strong>Target Date:</strong> ${forecastData.target_date || 'Feb 5, 2026'}</p>
                <p><strong>Risk Level:</strong> <span style="color: ${color}">${zone.risk_level.toUpperCase()}</span></p>
                <p><strong>Probability:</strong> ${(zone.probability * 100).toFixed(0)}%</p>
                <p><strong>Risk Score:</strong> ${zone.risk_score}/100</p>
                <p><strong>Based on:</strong> ${forecastData.based_on_dates || 'Recent data'}</p>
            </div>
        `;
        rect.bindPopup(popupContent);
    });
    
    console.log(`Visualized ${forecastData.risk_zones.length} FUTURE risk predictions for ${forecastData.target_date}`);
}

// Toggle spread visualization
async function toggleSpreadPredictions() {
    predictionState.showSpread = !predictionState.showSpread;
    
    const checkbox = document.getElementById('showSpread');
    if (checkbox) checkbox.checked = predictionState.showSpread;
    
    if (predictionState.showSpread && !spreadData) {
        await loadSpreadPredictions();
    }
    
    visualizeSpread();
}

// Toggle risk forecast visualization
async function toggleRiskForecast() {
    predictionState.showForecast = !predictionState.showForecast;
    
    const checkbox = document.getElementById('showForecast');
    if (checkbox) checkbox.checked = predictionState.showForecast;
    
    if (predictionState.showForecast && !forecastData) {
        await loadRiskForecast();
    }
    
    visualizeRiskForecast();
}

// Update all predictions
async function updatePredictions() {
    const updateBtn = document.getElementById('updatePredictions');
    if (updateBtn) {
        updateBtn.disabled = true;
        updateBtn.textContent = 'Updating...';
    }
    
    try {
        // Load all prediction data with timeout
        const loadPromise = Promise.all([
            loadFireTracking(),
            loadSpreadPredictions(),
            loadRiskForecast()
        ]);
        
        // Add 30 second timeout (ML training takes time)
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout')), 30000)
        );
        
        await Promise.race([loadPromise, timeoutPromise]);
        
        // Update visualizations if enabled
        if (predictionState.showSpread) visualizeSpread();
        if (predictionState.showForecast) visualizeRiskForecast();
        
        console.log('Predictions updated successfully');
    } catch (error) {
        console.error('Error updating predictions:', error);
        alert('Prediction update timed out. Try toggling checkboxes individually.');
    } finally {
        // Always reset button state
        if (updateBtn) {
            updateBtn.disabled = false;
            updateBtn.textContent = 'Update Predictions';
        }
    }
}

// ============= END PREDICTION FUNCTIONS =============

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Forest Fire Risk Detection System - Initializing...');
    
    // Initialize map
    initMap();
    
    // Add event listener to analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.addEventListener('click', analyzeFireRisk);
    
    // Add timeline control event listeners
    const playPauseBtn = document.getElementById('playPauseBtn');
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', togglePlayPause);
    }
    
    const timelineSlider = document.getElementById('timelineSlider');
    if (timelineSlider) {
        timelineSlider.addEventListener('input', (e) => {
            goToDate(e.target.value);
        });
    }
    
    const speedControl = document.getElementById('speedControl');
    if (speedControl) {
        speedControl.addEventListener('change', (e) => {
            changeSpeed(e.target.value);
        });
    }
    
    // Add prediction control event listeners
    const showSpreadCheckbox = document.getElementById('showSpread');
    if (showSpreadCheckbox) {
        showSpreadCheckbox.addEventListener('change', toggleSpreadPredictions);
    }
    
    const showForecastCheckbox = document.getElementById('showForecast');
    if (showForecastCheckbox) {
        showForecastCheckbox.addEventListener('change', toggleRiskForecast);
    }
    
    const updatePredictionsBtn = document.getElementById('updatePredictions');
    if (updatePredictionsBtn) {
        updatePredictionsBtn.addEventListener('click', updatePredictions);
    }
    
    console.log('Application ready');
});
