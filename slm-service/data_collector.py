"""
Data collection module for threshold training.
Collects uncertainty and rejection probability data during inference.
"""
import json
import os
from pathlib import Path
from typing import Optional

class DataCollector:
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize data collector.
        
        Args:
            data_file: Path to JSONL file to save data. If None, uses default location.
        """
        if data_file is None:
            # Default to slm-service directory
            data_file = Path(__file__).parent / "training_data.jsonl"
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
    def record_data_point(self, uncertainty: float, rejection_prob: float, 
                         y_d_lt_x_d: bool, session_id: str, draft_id: int):
        """
        Record a data point for training.
        
        Args:
            uncertainty: SLM uncertainty (u_t)
            rejection_prob: Rejection probability from LLM (max(1 - y_d/x_d, 0))
            y_d_lt_x_d: Whether y_d < x_d (for calculating Δ)
            session_id: Session ID for correlation
            draft_id: Draft token ID
        """
        data_point = {
            'uncertainty': float(uncertainty),
            'rejection_prob': float(rejection_prob),
            'y_d_lt_x_d': bool(y_d_lt_x_d),
            'session_id': str(session_id),
            'draft_id': int(draft_id)
        }
        
        # Append to JSONL file
        with open(self.data_file, 'a') as f:
            f.write(json.dumps(data_point) + '\n')
    
    def get_data_count(self) -> int:
        """Get number of data points collected so far."""
        if not self.data_file.exists():
            return 0
        with open(self.data_file, 'r') as f:
            return sum(1 for line in f if line.strip())
    
    def clear_data(self):
        """Clear collected data file."""
        if self.data_file.exists():
            self.data_file.unlink()



