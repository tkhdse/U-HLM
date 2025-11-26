"""
Train linear regression model for threshold calculation.

Fits: β = au + b
where:
  - β = rejection probability (max(1 - y_d/x_d, 0))
  - u = SLM uncertainty
  - a, b = learned parameters

Then calculates threshold:
  - Option 1 (Risk-Averse): u_th = -b/a (when β = 0)
  - Option 2 (Risk-prone): u_th = (Δ - b)/a where Δ = P(y_d < x_d)
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import argparse

def load_training_data(data_file):
    """Load training data from JSONL file."""
    data_points = []
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
    """
    Fit linear regression: β = au + b
    
    Returns:
        model: Fitted LinearRegression model
        a: slope coefficient
        b: intercept
        r2: R-squared score
    """
    if not data_points:
        raise ValueError("No data points to fit model")
    
    # Extract features (uncertainty) and targets (rejection_prob)
    X = np.array([[dp['uncertainty']] for dp in data_points])
    y = np.array([dp['rejection_prob'] for dp in data_points])
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    a = model.coef_[0]
    b = model.intercept_
    r2 = model.score(X, y)
    
    return model, a, b, r2

def calculate_threshold(a, b, delta, mode='risk-prone'):
    """
    Calculate threshold based on fitted model.
    
    Args:
        a: Slope coefficient
        b: Intercept
        delta: P(y_d < x_d)
        mode: 'risk-averse' or 'risk-prone'
    
    Returns:
        threshold value
    """
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

def main():

    script_dir = Path(__file__).parent
    default_data_file = script_dir / 'training_data.jsonl'
    default_output_file = script_dir / 'threshold_model.json'

    parser = argparse.ArgumentParser(description='Train threshold model from collected data')
    parser.add_argument('--data-file', type=str, default=str(default_data_file),
                        help='Path to training data JSONL file')
    parser.add_argument('--output-file', type=str, default=str(default_output_file),
                        help='Path to save model parameters')
    parser.add_argument('--mode', type=str, default='risk-prone', choices=['risk-averse', 'risk-prone'],
                        help='Threshold calculation mode (default: risk-prone)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum number of samples required (default: 100)')
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please collect data first using: python main.py --collect-data")
        return
    
    print(f"Loading training data from: {data_file}")
    data_points = load_training_data(data_file)
    print(f"Loaded {len(data_points)} data points")
    
    if len(data_points) < args.min_samples:
        print(f"Warning: Only {len(data_points)} data points, recommended at least {args.min_samples}")
        print("Consider collecting more data for better model accuracy")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Calculate Δ = P(y_d < x_d)
    delta = calculate_delta(data_points)
    print(f"\nΔ = P(y_d < x_d) = {delta:.4f}")
    
    # Fit linear model
    print("\nFitting linear regression: β = au + b")
    model, a, b, r2 = fit_linear_model(data_points)
    print(f"Model parameters:")
    print(f"  a (slope) = {a:.6f}")
    print(f"  b (intercept) = {b:.6f}")
    print(f"  R² = {r2:.4f}")
    
    # Calculate threshold
    threshold = calculate_threshold(a, b, delta, mode=args.mode)
    print(f"\nCalculated threshold (mode: {args.mode}):")
    print(f"  u_th = {threshold:.6f}")
    
    # Validate threshold
    if threshold < 0:
        print("Warning: Threshold is negative, which may not be meaningful")
        print("Consider using risk-averse mode or collecting more data")
    elif threshold > 1:
        print("Warning: Threshold is greater than 1, which is unusual")
        print("Uncertainty is normalized to [0, 1], so threshold > 1 means all tokens will be skipped")
    
    # Save model parameters
    output_file = Path(args.output_file)
    model_data = {
        'a': float(a),
        'b': float(b),
        'delta': float(delta),
        'threshold': float(threshold),
        'mode': args.mode,
        'r2': float(r2),
        'n_samples': len(data_points)
    }
    
    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\nModel saved to: {output_file}")
    print("\nYou can now use this threshold in threshold_calc.py")

if __name__ == "__main__":
    main()



