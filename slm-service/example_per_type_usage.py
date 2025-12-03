"""
Example script demonstrating how to use per-type threshold models.
This shows how to:
1. Load per-type thresholds
2. Get threshold for a specific prompt type
3. Use it in generation
"""
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import threshold_calc
import asyncio
from main import generate_response
import json

def show_trained_thresholds():
    """Display all trained per-type thresholds."""
    print("=" * 60)
    print("TRAINED PER-TYPE THRESHOLDS")
    print("=" * 60)
    
    # Get all type-specific thresholds
    thresholds = threshold_calc.get_all_type_thresholds()
    
    if not thresholds:
        print("❌ No per-type thresholds found.")
        print("   Run calibrate_by_type.py first to train per-type models.")
        return False
    
    print(f"\nFound {len(thresholds)} trained prompt types:\n")
    for prompt_type, threshold in sorted(thresholds.items()):
        print(f"  {prompt_type:25s}: {threshold:.6f}")
    
    # Get full model info
    models_info = threshold_calc.get_models_by_type_info()
    if models_info:
        print(f"\nGlobal fallback threshold: {models_info.get('global_fallback_threshold', 0.5):.6f}")
    
    print()
    return True

def demo_threshold_lookup():
    """Demo looking up thresholds for different prompt types."""
    print("=" * 60)
    print("DEMO: THRESHOLD LOOKUP BY TYPE")
    print("=" * 60)
    
    # Example prompt types
    test_types = [
        'classification',
        'summarization',
        'code-generation',
        'creative-writing',
        'unknown-type'  # Test fallback behavior
    ]
    
    print("\nLooking up thresholds for example types:\n")
    for prompt_type in test_types:
        threshold = threshold_calc.get_threshold_by_type(prompt_type)
        print(f"  {prompt_type:25s}: {threshold:.6f}")
    
    print()

async def demo_generation_with_type(prompt, prompt_type):
    """Demo generating with a specific prompt type threshold."""
    print("=" * 60)
    print(f"DEMO: GENERATION WITH TYPE-SPECIFIC THRESHOLD")
    print("=" * 60)
    print(f"\nPrompt type: {prompt_type}")
    print(f"Prompt: {prompt}\n")
    
    # Get threshold for this type
    threshold = threshold_calc.get_threshold_by_type(prompt_type)
    print(f"Using threshold: {threshold:.6f}\n")
    
    # Note: To actually use per-type thresholds in generation,
    # you would need to modify main.py to accept a threshold parameter
    # or pass the prompt_type to threshold_calc.get_threshold()
    
    print("(Generation would happen here with type-specific threshold)")
    print("(To fully integrate, modify main.py to pass prompt_type to threshold_calc)")

def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("PER-TYPE THRESHOLD DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # 1. Show all trained thresholds
    if not show_trained_thresholds():
        return
    
    input("Press Enter to continue to threshold lookup demo...")
    print()
    
    # 2. Demo threshold lookup
    demo_threshold_lookup()
    
    input("Press Enter to see example integration...")
    print()
    
    # 3. Show how to integrate
    print("=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)
    print("""
To integrate per-type thresholds into your generation pipeline:

1. Modify threshold_calc.get_threshold() to accept a prompt_type parameter:
   
   def get_threshold(prompt_type=None, use_default=False):
       if prompt_type:
           return get_threshold_by_type(prompt_type)
       # ... existing global threshold logic ...

2. Modify main.py to pass prompt_type through the call chain:
   
   async def generate_response(prompt, ..., prompt_type=None):
       # ...
       u_th = threshold_calc.get_threshold(prompt_type=prompt_type)
       # ...

3. When calling generate_response, specify the prompt type:
   
   await generate_response(
       "Summarize this article...",
       prompt_type="summarization"
   )

4. Or create a wrapper that automatically detects prompt type:
   
   def classify_prompt_type(prompt):
       # Use keyword matching or a classifier
       if "summarize" in prompt.lower():
           return "summarization"
       # ...
       
   prompt_type = classify_prompt_type(prompt)
   await generate_response(prompt, prompt_type=prompt_type)
""")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
Next steps:
1. Train per-type models: python calibrate_by_type.py
2. Integrate per-type thresholds into main.py
3. Optionally build a prompt type classifier for automatic type detection
""")

if __name__ == "__main__":
    main()

