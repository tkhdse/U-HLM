"""
End-to-end calibration script for U-HLM.
1. Reads prompts from a text file.
2. Runs inference to collect uncertainty vs. rejection data.
3. Trains a linear regression model.
4. Saves the optimal threshold configuration.
"""
import sys
import json
import asyncio
import argparse
import traceback
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from main import generate_response
from data_collector import DataCollector

# -----------------------------------------------------------------------------
# Training / Linear Regression Logic (Refactored from train_threshold.py)
# -----------------------------------------------------------------------------

def load_training_data(data_file):
    """Load training data from JSONL file."""
    data_points = []
    if not Path(data_file).exists():
        return []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                data_points.append(json.loads(line))
    return data_points

def calculate_delta(data_points):
    """Calculate Δ = P(y_d < x_d) from data."""
    if not data_points:
        return 0.0
    y_d_lt_x_d_count = sum(1 for dp in data_points if dp.get('y_d_lt_x_d', False))
    return y_d_lt_x_d_count / len(data_points)

def fit_linear_model(data_points):
    """Fit linear regression: β = au + b"""
    if not data_points:
        raise ValueError("No data points to fit model")
    
    X = np.array([[dp['uncertainty']] for dp in data_points])
    y = np.array([dp['rejection_prob'] for dp in data_points])
    
    model = LinearRegression()
    model.fit(X, y)
    
    a = model.coef_[0]
    b = model.intercept_
    r2 = model.score(X, y)
    
    return model, a, b, r2

def calculate_threshold(a, b, delta, mode='risk-prone'):
    """Calculate threshold based on fitted model."""
    if abs(a) < 1e-10:
        print("Warning: slope 'a' is very close to zero, cannot calculate threshold")
        return 0.0
    
    if mode == 'risk-averse':
        # Option 1: β = 0, so u_th = -b/a
        threshold = -b / a
    elif mode == 'risk-prone':
        # Option 2: β = Δ, so u_th = (Δ - b)/a
        threshold = (delta - b) / a
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return threshold

def train_and_save_model(data_file, output_file, mode='risk-prone'):
    """Run the training process and save the model."""
    print(f"\n📊 Starting Training Phase...")
    print(f"   Loading data from: {data_file}")
    
    data_points = load_training_data(data_file)
    if not data_points:
        print("Error: No data points found.")
        return False
        
    print(f"   Loaded {len(data_points)} data points")

    # Calculate Δ
    delta = calculate_delta(data_points)
    print(f"   Δ = P(y_d < x_d) = {delta:.4f}")

    # Fit model
    print("   Fitting linear regression...")
    try:
        _, a, b, r2 = fit_linear_model(data_points)
    except Exception as e:
        print(f"   Error fitting model: {e}")
        return False

    print(f"   Model: β = {a:.4f}u + {b:.4f} (R²={r2:.4f})")

    # Calculate threshold
    threshold = calculate_threshold(a, b, delta, mode)
    print(f"   Calculated Threshold ({mode}): {threshold:.6f}")

    # Save
    model_data = {
        'a': float(a),
        'b': float(b),
        'delta': float(delta),
        'threshold': float(threshold),
        'mode': mode,
        'r2': float(r2),
        'n_samples': len(data_points)
    }
    
    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"✅ Model saved to: {output_file}")
    return True

# -----------------------------------------------------------------------------
# Data Collection Logic
# -----------------------------------------------------------------------------

def load_prompts(prompt_file):
    """Read prompts from a text file, one per line."""
    path = Path(prompt_file)
    if not path.exists():
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

async def run_collection(prompts, max_tokens, use_chat_template, data_file):
    """Run data collection loop."""
    data_collector = DataCollector(data_file=str(data_file))
    print(f"\n🚀 Starting Data Collection Phase...")
    print(f"   Prompts to process: {len(prompts)}")
    print(f"   Output file: {data_file}")
    print("-" * 60)

    successful = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Processing: '{prompt}'")
        try:
            await generate_response(
                prompt,
                max_tokens=max_tokens,
                use_chat_template=use_chat_template,
                simulate_network=False,
                data_collector=data_collector
            )
            successful += 1
            print(f"   ✓ Done. Total points: {data_collector.get_data_count()}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
    
    print("-" * 60)
    print(f"Data collection complete. Collected {data_collector.get_data_count()} points.")
    return data_collector.get_data_count() > 0

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    default_data_file = script_dir / 'training_data.jsonl'
    default_model_file = script_dir / 'threshold_model.json'

    parser = argparse.ArgumentParser(description='U-HLM Calibration: Collect Data & Train Threshold')
    
    # Inputs
    parser.add_argument('prompt_file', type=str, help='Path to text file with prompts (one per line)')
    
    # Config
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens per prompt')
    parser.add_argument('--use-chat-template', action='store_true', help='Format prompts using chat template')
    parser.add_argument('--mode', type=str, default='risk-prone', choices=['risk-averse', 'risk-prone'], 
                        help='Threshold calculation mode')
    
    # Outputs
    parser.add_argument('--data-file', type=str, default=str(default_data_file), help='Path to save training data')
    parser.add_argument('--output-file', type=str, default=str(default_model_file), help='Path to save threshold model')
    
    # Skip options
    parser.add_argument('--skip-collection', action='store_true', help='Skip collection, only train on existing data')

    args = parser.parse_args()

    # 1. Data Collection
    if not args.skip_collection:
        prompts = load_prompts(args.prompt_file)
        if not prompts:
            return

        # Run async collection
        success = asyncio.run(run_collection(
            prompts, 
            args.max_tokens, 
            args.use_chat_template, 
            args.data_file
        ))
        
        if not success:
            print("Data collection failed or produced no data. Aborting training.")
            return
    else:
        print("Skipping data collection as requested.")

    # 2. Training
    train_and_save_model(args.data_file, args.output_file, args.mode)

if __name__ == "__main__":
    main()

