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
    python simulation.py

3. Edit parameters below to run different experiments. (In This project for the simulation reports and demonstration  we kept the other parameters constant and changed the developer and QA tester numbers for demonstrating their effects on average waiting time and average system time of the bug).

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




