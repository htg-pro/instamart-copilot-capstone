"""run_pipeline.py - Master execution script for the entire Instamart Copilot pipeline."""
import sys
import os
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from part_a.run_part_a import main as run_part_a
from part_b.run_part_b import main as run_part_b
from part_c.run_part_c import main as run_part_c
from part_d.run_part_d import main as run_part_d

def main():
    parser = argparse.ArgumentParser(description="Run Parts A-D of the offline Instamart capstone.")
    parser.add_argument("--generate", action="store_true", help="Regenerate the supplied synthetic data first")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if args.generate:
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    elif not (root / "data/raw/orders.csv").exists():
        parser.error("Raw data missing. Use --generate or run python generate_data.py first.")
    print("====================================================================")
    print("       INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT            ")
    print("                END-TO-END PIPELINE EXECUTION                       ")
    print("====================================================================\n")

    print("\n>>> EXECUTING PART A: DATA FOUNDATION (CLEAN, STORE, QUERY) <<<")
    run_part_a()

    print("\n>>> EXECUTING PART B: CUSTOMER SEGMENTS & LAPSE PREDICTION <<<")
    run_part_b()

    print("\n>>> EXECUTING PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS <<<")
    run_part_c()

    print("\n>>> EXECUTING PART D: POLICY RAG ASSISTANT & REFUND AGENT <<<")
    run_part_d()

    print("\n====================================================================")
    print("       ALL PIPELINE PARTS (A, B, C, D) COMPLETED SUCCESSFULLY!      ")
    print("====================================================================")

if __name__ == "__main__":
    main()
