"""run_pipeline.py - Master execution script for the entire Instamart Copilot pipeline."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from part_a.run_part_a import main as run_part_a
from part_b.run_part_b import main as run_part_b
from part_c.run_part_c import main as run_part_c
from part_d.run_part_d import main as run_part_d

def main():
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
