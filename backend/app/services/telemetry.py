import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def parse_telemetry(file_path: str) -> dict:
    """
    Parses the flight telemetry Excel/CSV file and returns a normalized dictionary.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # Standardize column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Required columns mapping
        # Map internal schema keys to possible CSV/Excel column names
        column_map = {
            "timestamp_date": ["LclDate", "Date"],
            "timestamp_time": ["LclTime", "Time"],
            "lat": ["Latitude", "Lat"],
            "lon": ["Longitude", "Lon", "Long"],
            "alt_msl": ["AltMSL", "Alt", "Altitude"],
            "alt_agl": ["AltAGL", "AGL"],
            "ias": ["IAS", "Indicated Airspeed"],
            "gnd_spd": ["GndSpd", "Ground Speed", "Speed"],
            "v_spd": ["VSpd", "Vertical Speed", "VS"],
            "pitch": ["Pitch"],
            "roll": ["Roll", "Bank"],
            "heading": ["HDG", "Heading"],
            "flaps": ["Flaps"],
            "rpm": ["E1RPM", "RPM", "Engine RPM"]
        }
        
        # Helper to find column
        def get_col(keys):
            for k in keys:
                if k in df.columns:
                    return k
            return None

        # Create normalized dataframe
        ndf = pd.DataFrame()
        
        # Parse Timestamp
        date_col = get_col(column_map["timestamp_date"])
        time_col = get_col(column_map["timestamp_time"])
        
        if date_col and time_col:
            # Combine Date and Time columns
            # Assuming format is standard, but might need robust parsing
            # For now, let's try to convert to string and combine
            
            # Function to combine date and time objects/strings
            def combine_datetime(row):
                d = row[date_col]
                t = row[time_col]
                
                # Handle various formats (string vs datetime objects)
                if isinstance(d, str):
                    d_str = d
                else:
                    d_str = d.strftime("%Y-%m-%d")
                    
                if isinstance(t, str):
                    t_str = t
                else:
                    t_str = t.strftime("%H:%M:%S")
                    
                return pd.to_datetime(f"{d_str} {t_str}")

            ndf["timestamp"] = df.apply(combine_datetime, axis=1)
        else:
            # Fallback: Generate relative time if no timestamp
            ndf["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df), freq="1S")

        # Calculate relative time in seconds
        start_time = ndf["timestamp"].iloc[0]
        ndf["time_sec"] = (ndf["timestamp"] - start_time).dt.total_seconds()

        # Map other columns
        mappings = [
            ("lat", "lat"), ("lon", "lon"), 
            ("alt_msl", "alt_msl"), ("alt_agl", "alt_agl"),
            ("ias", "ias"), ("gnd_spd", "gnd_spd"), ("v_spd", "v_spd"),
            ("pitch", "pitch"), ("roll", "roll"), ("heading", "heading"),
            ("flaps", "flaps"), ("rpm", "rpm")
        ]
        
        for schema_key, map_key in mappings:
            col_name = get_col(column_map[map_key])
            if col_name:
                ndf[schema_key] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
            else:
                ndf[schema_key] = 0.0 # Default to 0 if missing
                
        # Convert to list of dicts for JSON serialization
        # Replace NaN with None or 0 for JSON safety
        ndf = ndf.replace({np.nan: None})
        
        result = {
            "metadata": {
                "start_time": start_time.isoformat(),
                "duration_sec": ndf["time_sec"].iloc[-1],
                "point_count": len(ndf)
            },
            "data": ndf.to_dict(orient="records")
        }
        
        return result

    except Exception as e:
        print(f"Error parsing telemetry: {e}")
        raise e
