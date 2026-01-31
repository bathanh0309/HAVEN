"""
Configuration loader with environment variable interpolation.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class GeneralConfig(BaseModel):
    fps_limit: int = 30
    resize_width: int = 640
    log_level: str = "INFO"
    output_dir: str = "/mnt/user-data/outputs"


class CameraConfig(BaseModel):
    id: str
    name: str
    type: str  # "video" or "rtsp"
    source: str
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True


class InferenceConfig(BaseModel):
    yolo: Dict[str, Any]
    pose: Dict[str, Any]
    tracker: Dict[str, Any]


class ReIDConfig(BaseModel):
    enabled: bool = True
    face: Dict[str, Any]
    gait: Dict[str, Any]
    appearance: Dict[str, Any]
    thresholds: Dict[str, float]
    multi_prototype: Dict[str, Any]
    quality: Dict[str, Any]


class StorageConfig(BaseModel):
    db_path: str
    save_snapshots: bool = True
    snapshot_dir: str


class APIConfig(BaseModel):
    websocket: Dict[str, Any]
    mjpeg: Optional[Dict[str, Any]] = None


class Settings(BaseSettings):
    """Main settings class."""
    
    general: GeneralConfig
    cameras: List[CameraConfig]
    inference: InferenceConfig
    reid: ReIDConfig
    storage: StorageConfig
    api: APIConfig
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def interpolate_env_vars(value: Any) -> Any:
    """
    Replace ${VAR_NAME} with environment variable values.
    
    Example:
        "${RTSP_USERNAME}" -> "admin"
    """
    if isinstance(value, str):
        # Find all ${VAR} patterns
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return pattern.sub(replacer, value)
    
    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    
    return value


def load_config(config_path: str = "backend/config/sources.yaml") -> Settings:
    """
    Load configuration from YAML file with environment variable interpolation.
    
    Args:
        config_path: Path to sources.yaml
    
    Returns:
        Settings object
    """
    # Load environment variables
    # Check if we are in backend dir or root
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("backend/.env")
    
    if env_path.exists():
        load_dotenv(env_path)
    
    # Check config path
    if not os.path.exists(config_path):
         # try relative to backend
        alt_path = os.path.join("backend", config_path)
        if os.path.exists(alt_path):
             config_path = alt_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Interpolate environment variables
    config_dict = interpolate_env_vars(config_dict)
    
    # Parse with Pydantic
    settings = Settings(**config_dict)
    
    return settings


# Example usage
if __name__ == "__main__":
    try:
        config = load_config("backend/config/sources.example.yaml") # Use example for test
        print(f"Loaded config with {len(config.cameras)} cameras")
        for cam in config.cameras:
            print(f"  - {cam.name} ({cam.type}): {'enabled' if cam.enabled else 'disabled'}")
    except Exception as e:
        print(f"Error loading config: {e}")

