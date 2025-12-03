import json
import os
from typing import Dict, Any, Optional

def load_profiles() -> Dict[str, Any]:
    """
    Loads aircraft profiles from the JSON configuration file.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to app, then into config
    config_path = os.path.join(os.path.dirname(current_dir), "config", "aircraft_profiles.json")
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Aircraft profiles file not found at {config_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in aircraft profiles file at {config_path}")
        return {}

def get_aircraft_profile(plane_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the profile for a specific aircraft.
    Returns None if not found.
    """
    profiles = load_profiles()
    return profiles.get(plane_name)
