"""
Per-prompt-type calibration script for U-HLM.
1. Reads prompts from training_prompts.json (with labels).
2. Runs inference to collect uncertainty vs. rejection data per prompt type.
3. Trains separate linear regression models for each prompt type.
4. Saves per-type optimal threshold configuration.
"""
import sys
import json
import asyncio
import argparse
import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from main import generate_response
from data_collector import DataCollector

# -----------------------------------------------------------------------------
# Training / Linear Regression Logic (Per-Type)
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
        print(f"      Warning: slope 'a' is very close to zero, cannot calculate threshold")
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

def train_model_for_type(prompt_type, data_points, mode='risk-prone'):
    """Train model for a specific prompt type."""
    print(f"\n   📊 Training model for type: {prompt_type}")
    print(f"      Data points: {len(data_points)}")
    
    if len(data_points) < 5:
        print(f"      ⚠️  Too few data points ({len(data_points)}), skipping")
        return None
    
    # Calculate Δ
    delta = calculate_delta(data_points)
    print(f"      Δ = P(y_d < x_d) = {delta:.4f}")
    
    # Fit model
    try:
        _, a, b, r2 = fit_linear_model(data_points)
    except Exception as e:
        print(f"      ✗ Error fitting model: {e}")
        return None
    
    print(f"      Model: β = {a:.4f}u + {b:.4f} (R²={r2:.4f})")
    
    # Calculate threshold
    threshold = calculate_threshold(a, b, delta, mode)
    print(f"      Threshold ({mode}): {threshold:.6f}")
    
    return {
        'a': float(a),
        'b': float(b),
        'delta': float(delta),
        'threshold': float(threshold),
        'mode': mode,
        'r2': float(r2),
        'n_samples': len(data_points)
    }

# -----------------------------------------------------------------------------
# Data Collection Logic (Per-Type)
# -----------------------------------------------------------------------------

def load_prompts_from_json(json_file):
    """Load prompts from training_prompts.json."""
    path = Path(json_file)
    if not path.exists():
        print(f"Error: Prompt file '{json_file}' not found.")
        return {}
    
    with open(path, 'r') as f:
        prompts = json.load(f)
    
    # Group by label
    prompts_by_type = defaultdict(list)
    for item in prompts:
        prompt_type = item.get('label', 'unknown')
        prompts_by_type[prompt_type].append(item['prompt'])
    
    return dict(prompts_by_type)

async def run_collection_for_type(prompt_type, prompts, max_tokens, use_chat_template, data_file):
    """Run data collection for a specific prompt type."""
    data_collector = DataCollector(data_file=str(data_file))
    print(f"\n   🚀 Collecting data for type: {prompt_type}")
    print(f"      Prompts: {len(prompts)}")
    print(f"      Output: {data_file.name}")
    
    successful = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"      [{i}/{len(prompts)}] '{prompt[:50]}...'")
        try:
            await generate_response(
                prompt,
                max_tokens=max_tokens,
                use_chat_template=use_chat_template,
                simulate_network=False,
                data_collector=data_collector
            )
            successful += 1
            print(f"         ✓ Done. Total points: {data_collector.get_data_count()}")
        except Exception as e:
            print(f"         ✗ Error: {e}")
            # Don't print full traceback to keep output cleaner
    
    total_points = data_collector.get_data_count()
    print(f"      ✅ Complete. Collected {total_points} data points.")
    return total_points

async def run_collection_all_types(prompts_by_type, max_tokens, use_chat_template, data_dir):
    """Run data collection for all prompt types."""
    print(f"\n{'='*60}")
    print(f"📦 DATA COLLECTION PHASE")
    print(f"{'='*60}")
    print(f"Prompt types: {len(prompts_by_type)}")
    print(f"Data directory: {data_dir}")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for prompt_type, prompts in prompts_by_type.items():
        data_file = data_dir / f"training_data_{prompt_type}.jsonl"
        
        # Clear old data file
        if data_file.exists():
            data_file.unlink()
        
        total_points = await run_collection_for_type(
            prompt_type, 
            prompts, 
            max_tokens, 
            use_chat_template, 
            data_file
        )
        
        results[prompt_type] = {
            'data_file': str(data_file),
            'num_prompts': len(prompts),
            'data_points': total_points
        }
    
    return results

# -----------------------------------------------------------------------------
# Training Logic (All Types)
# -----------------------------------------------------------------------------

def train_all_types(data_dir, mode='risk-prone'):
    """Train models for all prompt types."""
    print(f"\n{'='*60}")
    print(f"🎓 TRAINING PHASE")
    print(f"{'='*60}")
    
    data_files = list(data_dir.glob("training_data_*.jsonl"))
    print(f"Found {len(data_files)} data files")
    
    models_by_type = {}
    
    for data_file in data_files:
        # Extract prompt type from filename
        # training_data_classification.jsonl -> classification
        prompt_type = data_file.stem.replace('training_data_', '')
        
        print(f"\n📊 Processing: {prompt_type}")
        print(f"   Data file: {data_file.name}")
        
        # Load data
        data_points = load_training_data(data_file)
        if not data_points:
            print(f"   ⚠️  No data points found, skipping")
            continue
        
        # Train model
        model_data = train_model_for_type(prompt_type, data_points, mode)
        if model_data:
            models_by_type[prompt_type] = model_data
        else:
            print(f"   ⚠️  Training failed for {prompt_type}")
    
    return models_by_type

def save_models(models_by_type, output_file, global_fallback_threshold=0.5):
    """Save all models to a single JSON file."""
    print(f"\n{'='*60}")
    print(f"💾 SAVING MODELS")
    print(f"{'='*60}")
    
    output_data = {
        'models_by_type': models_by_type,
        'global_fallback_threshold': global_fallback_threshold,
        'prompt_types': list(models_by_type.keys()),
        'total_types': len(models_by_type)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved {len(models_by_type)} models to: {output_file}")
    print(f"\nThresholds by type:")
    for prompt_type, model_data in sorted(models_by_type.items()):
        print(f"  {prompt_type:25s}: {model_data['threshold']:.6f} (R²={model_data['r2']:.4f}, n={model_data['n_samples']})")

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    default_data_dir = script_dir / 'training_data_by_type'
    default_model_file = script_dir / 'threshold_models_by_type.json'
    default_prompt_file = script_dir / 'training_prompts.json'

    parser = argparse.ArgumentParser(
        description='U-HLM Per-Type Calibration: Collect Data & Train Type-Specific Thresholds'
    )
    
    # Inputs
    parser.add_argument('--prompt-file', type=str, default=str(default_prompt_file),
                       help='Path to JSON file with prompts and labels')
    
    # Config
    parser.add_argument('--max-tokens', type=int, default=50, 
                       help='Max tokens per prompt (default: 50)')
    parser.add_argument('--use-chat-template', action='store_true', 
                       help='Format prompts using chat template')
    parser.add_argument('--mode', type=str, default='risk-prone', 
                       choices=['risk-averse', 'risk-prone'], 
                       help='Threshold calculation mode (default: risk-prone)')
    parser.add_argument('--global-fallback', type=float, default=0.5,
                       help='Global fallback threshold for unknown types (default: 0.5)')
    
    # Outputs
    parser.add_argument('--data-dir', type=str, default=str(default_data_dir), 
                       help='Directory to save per-type training data')
    parser.add_argument('--output-file', type=str, default=str(default_model_file), 
                       help='Path to save threshold models')
    
    # Skip options
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip collection, only train on existing data')
    parser.add_argument('--types', type=str, nargs='+',
                       help='Only process specific prompt types (default: all)')

    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)

    # 1. Data Collection
    if not args.skip_collection:
        print(f"\n{'='*60}")
        print(f"🚀 U-HLM PER-TYPE CALIBRATION")
        print(f"{'='*60}")
        
        # Load prompts
        prompts_by_type = load_prompts_from_json(args.prompt_file)
        if not prompts_by_type:
            print("Error: No prompts loaded")
            return
        
        # Filter types if specified
        if args.types:
            prompts_by_type = {
                k: v for k, v in prompts_by_type.items() 
                if k in args.types
            }
            if not prompts_by_type:
                print(f"Error: No prompts found for types: {args.types}")
                return
        
        print(f"\nPrompt types to process:")
        for prompt_type, prompts in sorted(prompts_by_type.items()):
            print(f"  {prompt_type:25s}: {len(prompts)} prompts")
        
        # Run collection
        collection_results = asyncio.run(run_collection_all_types(
            prompts_by_type,
            args.max_tokens,
            args.use_chat_template,
            data_dir
        ))
        
        print(f"\n{'='*60}")
        print("Collection Summary:")
        for prompt_type, result in collection_results.items():
            print(f"  {prompt_type:25s}: {result['data_points']} points from {result['num_prompts']} prompts")
    else:
        print("⏭️  Skipping data collection as requested.")
    
    # 2. Training
    models_by_type = train_all_types(data_dir, args.mode)
    
    if not models_by_type:
        print("\n❌ No models were successfully trained. Exiting.")
        return
    
    # 3. Save
    save_models(models_by_type, args.output_file, args.global_fallback)
    
    print(f"\n{'='*60}")
    print("✅ Calibration complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

