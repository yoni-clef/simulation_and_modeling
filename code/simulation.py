"""
bug_fix_simulation.py

Discrete-event style simulation of a Bug Fix Cycle (no SimPy).
Produces:
 - metrics.csv        : per-replication metrics
 - avg_backlog.png    : average backlog over time (aggregated)
 - wait_hist.png      : histogram of avg waiting times across replications
 - report.md          : short markdown report with parameters and summary

How to use:
 1. Install dependencies (if needed):
    pip install numpy pandas matplotlib

 2. Run:
    python bug_fix_simulation.py

3. Edit parameters below to run different experiments.
"""

import random
import statistics
import os
import math
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Simulation parameters (edit)
# -----------------------------
HOURS_PER_DAY = 8.0         # working hours per day
SIM_DAYS = 30               # simulation length per replication (days)
SIM_TIME = SIM_DAYS * HOURS_PER_DAY  # total simulation time in hours
REPLICATIONS = 50           # number of independent replications

arrival_rate_per_day = 10.0   # bugs arriving per day (lambda)
arrival_rate = arrival_rate_per_day / HOURS_PER_DAY  # per hour

dev_mean_fix_hours = 2.0      # mean (hours) to fix a bug (exponential)
qa_mean_hours = 1.0           # mean (hours) for QA testing (exponential)
rework_prob = 0.20            # probability QA returns bug to developers (feedback)

num_developers = 1           # number of developers (server capacity)
num_qa = 1              # number of QA testers

SAMPLE_INTERVAL = 0.5         # hours; backlog sampling granularity

RANDOM_SEED_BASE = 1000       # base seed for reproducibility per replication

OUT_DIR = "simulation_outputs"  # where outputs are saved
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Utility samplers
# -----------------------------
def exp_sample(mean):
    """Exponential variate with given mean."""
    if mean <= 0:
        return 0.0
    return random.expovariate(1.0 / mean)

# -----------------------------
# Core simulation for one replication
# -----------------------------
def generate_arrivals(seed):
    "Generate arrival times (in hours) within SIM_TIME using Poisson process (exponential interarrival)."
    random.seed(seed)
    t = 0.0
    arrivals = []
    while True:
        inter = random.expovariate(arrival_rate)  # interarrival in hours
        t += inter
        if t >= SIM_TIME:
            break
        arrivals.append(t)
    return arrivals

def simulate_replication(seed):
    """
    Simulate one replication.
    Approach:- For each arrival, simulate its lifecycle (dev fix(s) + QA(s)) deterministically against server next-available times.
    We track:
      - waiting times experienced at dev queue (including rework waits)
      - total system time per bug (arrival -> final close)
      - throughput (closed bugs)
      - occupancy intervals for backlog sampling
    """
    random.seed(seed)
    arrivals = generate_arrivals(seed)

    # Next available time for each dev and QA server (initially 0 = free at time 0)
    dev_next = [0.0 for _ in range(num_developers)]
    qa_next = [0.0 for _ in range(num_qa)]

    waiting_times = []   # list of waiting durations at developer queue (includes rework waits)
    system_times = []    # total time in system per closed bug (hours)
    occupancy_intervals = []  # list of (start, end) for each bug lifecycle in system
    throughput = 0

    for arrival_time in arrivals:
        # track the earliest time the bug leaves the system
        current_time = arrival_time

        # For occupancy interval start, it's the arrival_time
        lifecycle_start = arrival_time

        # --- Developer service(s) + QA(s) loop (handles rework feedback) ---
        while True:
            # Find a dev server: if any dev is free at current_time, start immediately; else wait until earliest free dev
            free_dev_index = None
            for i in range(num_developers):
                if dev_next[i] <= current_time:
                    free_dev_index = i
                    break
            if free_dev_index is None:
                # find earliest freeing dev
                free_dev_index = min(range(num_developers), key=lambda i: dev_next[i])
                dev_start = dev_next[free_dev_index]
                wait = dev_start - current_time
                # This wait is time spent waiting for dev (for initial fix or rework)
                waiting_times.append(wait)
                current_time = dev_start
            else:
                # immediate service
                waiting_times.append(0.0)

            # dev service time
            dev_service = exp_sample(dev_mean_fix_hours)
            dev_next[free_dev_index] = current_time + dev_service
            current_time = dev_next[free_dev_index]

            # QA: find QA server (same logic)
            free_qa_index = None
            for j in range(num_qa):
                if qa_next[j] <= current_time:
                    free_qa_index = j
                    break
            if free_qa_index is None:
                free_qa_index = min(range(num_qa), key=lambda j: qa_next[j])
                qa_start = qa_next[free_qa_index]
                current_time = qa_start  # wait for QA
            # QA service
            qa_service = exp_sample(qa_mean_hours)
            qa_next[free_qa_index] = current_time + qa_service
            current_time = qa_next[free_qa_index]

            # QA decision: accept or return for rework
            if random.random() < rework_prob:
                # returned to developers: loop again with current_time as arrival for rework
                # Note: waiting_times records are appended in the next loop iteration.
                continue
            else:
                # accepted and closed
                throughput += 1
                lifecycle_end = current_time
                system_times.append(lifecycle_end - lifecycle_start)
                occupancy_intervals.append((lifecycle_start, lifecycle_end))
                break

    # Build backlog time series by sampling occupancy intervals
    sample_times = [i * SAMPLE_INTERVAL for i in range(int(math.ceil(SIM_TIME / SAMPLE_INTERVAL)) + 1)]
    backlog_counts = []
    for t in sample_times:
        count = sum(1 for (s, e) in occupancy_intervals if s <= t < e)
        backlog_counts.append(count)

    result = {
        "waiting_times": waiting_times,
        "system_times": system_times,
        "throughput": throughput,
        "sample_times": sample_times,
        "backlog_counts": backlog_counts
    }
    return result

# -----------------------------
# Run multiple replications
# -----------------------------
def run_experiments():
    all_rep_metrics = []
    all_backlogs = []  # list of (sample_times, backlog_counts) per replication

    for rep in range(REPLICATIONS):
        seed = RANDOM_SEED_BASE + rep
        res = simulate_replication(seed)
        mean_wait = statistics.mean(res["waiting_times"]) if res["waiting_times"] else 0.0
        mean_system = statistics.mean(res["system_times"]) if res["system_times"] else 0.0
        throughput_per_day = res["throughput"] / SIM_DAYS  # normalized per day

        rep_row = {
            "replication": rep + 1,
            "mean_wait_hours": mean_wait,
            "mean_system_hours": mean_system,
            "throughput_per_day": throughput_per_day,
            "closed_bugs_total": res["throughput"]
        }
        all_rep_metrics.append(rep_row)
        all_backlogs.append((res["sample_times"], res["backlog_counts"]))

    df_metrics = pd.DataFrame(all_rep_metrics)
    return df_metrics, all_backlogs

# -----------------------------
# Postprocessing & plotting
# -----------------------------
def aggregate_and_plot(df_metrics, all_backlogs):
    # Summary statistics
    summary = {
        "avg_wait_hours_mean": df_metrics["mean_wait_hours"].mean(),
        "avg_wait_hours_stdev": df_metrics["mean_wait_hours"].std(ddof=0) if len(df_metrics) > 1 else 0.0,
        "avg_system_hours_mean": df_metrics["mean_system_hours"].mean(),
        "avg_throughput_per_day_mean": df_metrics["throughput_per_day"].mean(),
        # approximate utilizations via traffic intensity:
        "dev_util_estimate": min((arrival_rate * dev_mean_fix_hours) / num_developers, 1.0),
        "qa_util_estimate": min((arrival_rate * dev_mean_fix_hours) / num_developers * qa_mean_hours, 1.0)
    }

    # Save metrics CSV
    metrics_csv_path = os.path.join(OUT_DIR, "metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)

    # Aggregate backlog: compute mean backlog at each sample index
    sample_times = all_backlogs[0][0]
    mean_backlog = []
    for i in range(len(sample_times)):
        vals = [b[1][i] for b in all_backlogs if i < len(b[1])]
        mean_backlog.append(statistics.mean(vals) if vals else 0.0)

    # Plot average backlog over time
    plt.figure(figsize=(10, 4))
    plt.plot(sample_times, mean_backlog)
    plt.xlabel("Time (hours)")
    plt.ylabel("Average backlog (bugs)")
    plt.title("Average Backlog over Time")
    plt.grid(True)
    plt.tight_layout()
    backlog_path = os.path.join(OUT_DIR, "avg_backlog.png")
    plt.savefig(backlog_path)
    plt.close()

    # Histogram of average waiting times across replications
    plt.figure(figsize=(8, 4))
    plt.hist(df_metrics["mean_wait_hours"], bins=15)
    plt.xlabel("Average waiting time (hours)")
    plt.ylabel("Frequency")
    plt.title("Distribution of avg waiting times (replications)")
    plt.tight_layout()
    wait_hist_path = os.path.join(OUT_DIR, "wait_hist.png")
    plt.savefig(wait_hist_path)
    plt.close()

    # Create a short markdown report
    report_lines = []
    report_lines.append("# Bug Fix Cycle Simulation Report\n")
    report_lines.append("**Parameters**\n")
    report_lines.append(f"- Arrival rate (bugs/day): {arrival_rate_per_day}\n")
    report_lines.append(f"- Developers (servers): {num_developers}\n")
    report_lines.append(f"- QA testers (servers): {num_qa}\n")
    report_lines.append(f"- Mean dev fix time (hours): {dev_mean_fix_hours}\n")
    report_lines.append(f"- Mean QA testing time (hours): {qa_mean_hours}\n")
    report_lines.append(f"- Rework probability: {rework_prob}\n")
    report_lines.append(f"- Simulation length: {SIM_DAYS} days ({SIM_TIME} hours)\n")
    report_lines.append(f"- Replications: {REPLICATIONS}\n")
    report_lines.append("\n**Summary (averaged over replications)**\n")
    report_lines.append(f"- Average waiting time (hours): {summary['avg_wait_hours_mean']:.3f} (stdev {summary['avg_wait_hours_stdev']:.3f})\n")
    report_lines.append(f"- Average system time (hours): {summary['avg_system_hours_mean']:.3f}\n")
    report_lines.append(f"- Average throughput (bugs/day): {summary['avg_throughput_per_day_mean']:.3f}\n")
    report_lines.append(f"- Approx. developer utilization (ρ): {summary['dev_util_estimate']:.3f}\n")
    report_lines.append(f"- Approx. QA utilization (ρ_qa): {summary['qa_util_estimate']:.3f}\n")
    report_lines.append("\n**Generated files**\n")
    report_lines.append(f"- {metrics_csv_path}\n")
    report_lines.append(f"- {backlog_path}\n")
    report_lines.append(f"- {wait_hist_path}\n")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.writelines("\n".join(report_lines))

    return {
        "metrics_csv": metrics_csv_path,
        "avg_backlog_png": backlog_path,
        "wait_hist_png": wait_hist_path,
        "report_md": report_path,
        "summary": summary
    }

# -----------------------------
# Main
# -----------------------------
def main():
    print("Running Bug Fix Cycle simulation")
    print(f"Parameters: arrival_rate_per_day={arrival_rate_per_day}, devs={num_developers}, qa={num_qa}, rework_prob={rework_prob}")
    df_metrics, all_backlogs = run_experiments()
    outputs = aggregate_and_plot(df_metrics, all_backlogs)

    # Print summary
    print("\nSummary metrics (approx):")
    for k, v in outputs["summary"].items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")

    print("\nFiles saved to directory:", OUT_DIR)
    for k, v in outputs.items():
        if k != "summary":
            print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
