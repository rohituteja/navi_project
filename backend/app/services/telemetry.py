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
            # G1000 files often have # metadata lines at the top
            df = pd.read_csv(file_path, comment='#')
        else:
            df = pd.read_excel(file_path)
            
        # Standardize column names (remove whitespace and internal spaces)
        # This helps normalize "Lcl Date" to "LclDate" and "E1 RPM" to "E1RPM"
        df.columns = df.columns.str.strip().str.replace(" ", "")
        
        # Required columns mapping (supporting both G3X and G1000 standardized names)
        column_map = {
            "timestamp_date": "LclDate",
            "timestamp_time": "LclTime",
            "lat": "Latitude",
            "lon": "Longitude",
            "alt_msl": "AltMSL",
            "alt_agl": ["AGL", "AltAGL"], # Check both
            "ias": "IAS",
            "gnd_spd": "GndSpd",
            "v_spd": "VSpd",
            "pitch": "Pitch",
            "roll": "Roll",
            "heading": "HDG",
            "track": "TRK",
            "flaps": "Flaps",
            "rpm": "E1RPM"
        }
        
        # Create normalized dataframe
        ndf = pd.DataFrame()
        
        # Parse Timestamp
        # Combine Date and Time columns
        def combine_datetime(row):
            date_col = column_map["timestamp_date"]
            time_col = column_map["timestamp_time"]
            
            if date_col not in row or time_col not in row:
                return datetime.now()
                
            d = row[date_col]
            t = row[time_col]
            
            # Handle various formats (string vs datetime objects)
            if isinstance(d, str):
                d_str = d.strip()
            elif pd.isna(d):
                d_str = "2000-01-01" # Default
            else:
                d_str = d.strftime("%Y-%m-%d")
                
            if isinstance(t, str):
                t_str = t.strip()
            elif pd.isna(t):
                t_str = "00:00:00" # Default
            else:
                t_str = t.strftime("%H:%M:%S")
                
            try:
                return pd.to_datetime(f"{d_str} {t_str}")
            except:
                return datetime.now()

        ndf["timestamp"] = df.apply(combine_datetime, axis=1)

        # Calculate relative time in seconds
        if not ndf.empty:
            start_time = ndf["timestamp"].iloc[0]
            ndf["time_sec"] = (ndf["timestamp"] - start_time).dt.total_seconds()
        else:
            start_time = datetime.now()
            ndf["time_sec"] = 0.0

        # Map other columns
        mappings = [
            ("lat", "lat"), ("lon", "lon"), 
            ("alt_msl", "alt_msl"), ("alt_agl", "alt_agl"),
            ("ias", "ias"), ("gnd_spd", "gnd_spd"), ("v_spd", "v_spd"),
            ("vertical_speed_fpm", "v_spd"), # Alias for clarity
            ("pitch", "pitch"), ("roll", "roll"), 
            ("heading", "heading"), ("track", "track"),
            ("flaps", "flaps"), ("rpm", "rpm")
        ]
        
        for schema_key, map_key in mappings:
            col_target = column_map[map_key]
            
            # Find the best column name
            found_col = None
            if isinstance(col_target, list):
                for candidate in col_target:
                    if candidate in df.columns:
                        found_col = candidate
                        break
            else:
                if col_target in df.columns:
                    found_col = col_target
            
            if found_col:
                ndf[schema_key] = pd.to_numeric(df[found_col], errors='coerce').fillna(0)
            else:
                ndf[schema_key] = 0.0 # Default to 0 if missing

        # AGL Fallback: If AGL is missing, use AltMSL relative to start
        if not ndf.empty and "alt_agl" in ndf.columns:
            if (ndf["alt_agl"] == 0).all() and not (ndf["alt_msl"] == 0).all():
                initial_msl = ndf["alt_msl"].iloc[0]
                ndf["alt_agl"] = ndf["alt_msl"] - initial_msl
                ndf["alt_agl"] = ndf["alt_agl"].clip(lower=0) # Clamp negative values
                
        # Convert to list of dicts for JSON serialization
        # Replace NaN with None or 0 for JSON safety
        ndf = ndf.replace({np.nan: None})

        # Calculate derived metrics
        # 1. Turn Rate (deg/sec)
        # Calculate difference in heading, handling 360-0 wrap
        ndf["heading_diff"] = ndf["heading"].diff().fillna(0)
        # Adjust for wrap-around (e.g., 359 -> 1 should be +2, not -358)
        ndf["heading_diff"] = ndf["heading_diff"].apply(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))
        
        # Time difference
        ndf["time_diff"] = ndf["time_sec"].diff().fillna(1) # Avoid div by zero
        ndf["turn_rate"] = ndf["heading_diff"] / ndf["time_diff"]
        
        # Smooth turn rate
        ndf["turn_rate"] = ndf["turn_rate"].rolling(window=3, center=True).mean().fillna(0)

        # 2. Is Ground
        # Ground speed < 65 knots AND Alt AGL < 50
        # Increased from 35kts to 65kts to ensure takeoff roll is classified as ground
        ndf["is_ground"] = (ndf["gnd_spd"] < 65) & (ndf["alt_agl"] < 50)

        
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
