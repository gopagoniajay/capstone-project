import argparse
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_loso import main as run_loso
from personalize import main as run_personalization
# from train import main as run_random_split # Assuming standard train exists or implement here

def run_evaluation(args):
    os.makedirs('results', exist_ok=True)
    
    if args.task == 'all' or args.task == 'loso':
        print("\n=== Running LOSO Evaluation ===")
        # We need to modify train_loso.py to accept arguments or just run it as is if it saves to the right place
        # For now, we assume the modified train_loso.py will save to results/loso_details.json
        run_loso()
        
    if args.task == 'all' or args.task == 'personalization':
        print("\n=== Running Personalization Evaluation ===")
        # We need to modify personalize.py to accept arguments
        run_personalization()
        
    if args.task == 'all' or args.task == 'random_split':
        print("\n=== Running Random Split 80/20 Evaluation ===")
        # This part might need a dedicated function if not existing
        # For now, placeholder or we can implement a quick training run here
        print("Random split evaluation not yet fully integrated. Using Mock data for now.")
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation to generate plot data")
    parser.add_argument('--task', type=str, default='all', choices=['all', 'loso', 'personalization', 'random_split'],
                        help="Which evaluation task to run")
    parser.add_argument('--fast', action='store_true', help="Run in fast mode (fewer epochs) for testing")
    
    args = parser.parse_args()
    
    run_evaluation(args)
