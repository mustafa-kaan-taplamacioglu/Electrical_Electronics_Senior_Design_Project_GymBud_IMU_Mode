"""
Global state management for API server
"""

from typing import Optional, Dict
import asyncio
from openai import OpenAI

# Import optional dependencies
try:
    from dataset_collector import DatasetCollector
    DATASET_COLLECTION_ENABLED = True
except ImportError:
    DATASET_COLLECTION_ENABLED = False
    DatasetCollector = None

try:
    from imu_dataset_collector import IMUDatasetCollector
    IMU_DATASET_COLLECTION_ENABLED = True
except ImportError:
    IMU_DATASET_COLLECTION_ENABLED = False
    IMUDatasetCollector = None

try:
    from dataset_tracker import DatasetTracker
    DATASET_TRACKER_ENABLED = True
except ImportError:
    DATASET_TRACKER_ENABLED = False
    DatasetTracker = None

# Global state
sessions = {}
connected_clients = set()
openai_client: Optional[OpenAI] = None
camera_training_collectors: Dict[str, DatasetCollector] = {} if DatasetCollector else {}
imu_training_collectors: Dict[str, IMUDatasetCollector] = {} if IMUDatasetCollector else {}
imu_bridge_tasks: Dict[str, asyncio.Task] = {}
dataset_tracker = DatasetTracker() if DATASET_TRACKER_ENABLED and DatasetTracker else None
dataset_collector = DatasetCollector("dataset") if DATASET_COLLECTION_ENABLED and DatasetCollector else None

