"""
Forest Fire Risk Detection System - Flask Backend
Uses NASA FIRMS MODIS data to classify and visualize fire risk
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Configuration
CSV_FILE = 'fire_nrt_M-C61_565334.csv'
MAX_ROWS = 10000  # Load more rows to ensure we get multiple dates
SAMPLE_SIZE = 5000  # Final sample size after date-based sampling

def load_fire_data():
    """Load and process fire data from CSV"""
    try:
        # Read CSV file with required columns including temporal data
        # Load more rows initially to ensure date diversity
        df = pd.read_csv(
            CSV_FILE,
            usecols=['latitude', 'longitude', 'brightness', 'acq_date', 'acq_time']
        )
        
        # Remove any NaN values
        df = df.dropna()
        
        # Sort by date and time chronologically
        df = df.sort_values(['acq_date', 'acq_time'])
        
        # Sample data to get representation across all dates
        # Group by date and sample proportionally
        unique_dates = df['acq_date'].unique()
        samples_per_date = max(10, SAMPLE_SIZE // len(unique_dates))
        
        sampled_dfs = []
        for date in unique_dates:
            date_df = df[df['acq_date'] == date]
            sample_count = min(len(date_df), samples_per_date)
            sampled_dfs.append(date_df.sample(n=sample_count, random_state=42))
        
        df = pd.concat(sampled_dfs).sort_values(['acq_date', 'acq_time'])
        
        # Limit to max sample size
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).sort_values(['acq_date', 'acq_time'])
        
        print(f"Loaded {len(df)} fire detections across {len(unique_dates)} dates")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def classify_risk(brightness):
    """
    Classify fire risk based on brightness temperature
    
    Args:
        brightness (float): Temperature in Kelvin
        
    Returns:
        str: Risk level (High, Medium, or Low)
    """
    if brightness > 330:
        return "High"
    elif brightness >= 300:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    """
    API endpoint to analyze fire risk data
    
    Returns:
        JSON: List of fire data points with risk classification
    """
    try:
        # Load fire data
        df = load_fire_data()
        
        if df is None or df.empty:
            return jsonify({
                'error': 'Unable to load fire data',
                'data': []
            }), 500
        
        # Apply risk classification
        df['risk'] = df['brightness'].apply(classify_risk)
        
        # Format date and time fields
        df['date'] = df['acq_date'].astype(str)
        df['time'] = df['acq_time'].astype(str).str.zfill(4)
        
        # Calculate timeline metadata
        unique_dates = sorted(df['date'].unique())
        timeline_metadata = {
            'start_date': unique_dates[0] if len(unique_dates) > 0 else None,
            'end_date': unique_dates[-1] if len(unique_dates) > 0 else None,
            'total_days': len(unique_dates),
            'dates': unique_dates
        }
        
        # Convert to list of dictionaries
        fire_data = df.to_dict('records')
        
        # Add statistics
        risk_counts = df['risk'].value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'total_points': len(fire_data),
            'timeline': timeline_metadata,
            'statistics': {
                'high_risk': risk_counts.get('High', 0),
                'medium_risk': risk_counts.get('Medium', 0),
                'low_risk': risk_counts.get('Low', 0)
            },
            'data': fire_data
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'data': []
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'csv_exists': os.path.exists(CSV_FILE)
    })

if __name__ == '__main__':
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"Warning: CSV file '{CSV_FILE}' not found!")
    else:
        print(f"CSV file found: {CSV_FILE}")
    
    # Run the Flask app
    print("Starting Forest Fire Risk Detection System...")
    print("Access the dashboard at: http://127.0.0.1:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
