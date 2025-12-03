# simulation_and_modeling

Discrete-event style simulation of a Bug Fix Cycle (no SimPy).
Produces:
 - metrics.csv        : per-replication metrics
 - avg_backlog.png    : average backlog over time (aggregated)
 - wait_hist.png      : histogram of avg waiting times across replications
 - report.md          : short markdown report with parameters and summary

How to use:
 1. Install dependencies (if needed):
    pip install numpy,pandas,matplotlib

 2. Run:
    python bug_fix_simulation.py

3. Edit parameters below to run different experiments.

