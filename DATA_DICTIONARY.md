# Synthetic vessel visits: data dictionary

One row is one visit to a fictional port. Times are UTC. The exported dataset has 500 visits; 150 earlier simulated visits warm up the queues and historical statistics and are not exported. No downloaded AIS rows are relabelled or mixed into this dataset.

| Column | Type / units | Definition | Model use |
|---|---|---|---|
| visit_id | string | Unique visit key, independent of vessel identity | Audit only |
| port_id | string | PORT_A, PORT_B, or PORT_C | Analysis only |
| arrival_time | UTC timestamp | Entry into the port's designated waiting area | Chronological ordering |
| berth_start_time | UTC timestamp | Start of berth occupancy | Outcome only |
| service_end_time | UTC timestamp | End of simulated berth service | Outcome/audit only |
| label_available_time | UTC timestamp | Berth start for a wait <=6h; arrival +6h +1 microsecond for a longer wait | Cutoff eligibility only |
| queue_length | integer >=0 | Other vessels waiting at this port, before this arrival joins | Derive queue_per_berth |
| operational_berths | positive integer | Fixed capacity for this fictional port | Derive normalized inputs |
| busy_berths | integer, 0 to capacity | Vessels currently occupying berths | Derive berth_utilization |
| arrivals_past_6h | integer >=0 | Prior arrivals at the same port in [arrival -6h, arrival) | Derive arrival_pressure |
| weather_severity | 0, 1, 2, or missing | Clear, moderate, or severe weather at arrival | Input |
| past_avg_service_hours | positive hours or missing | Mean service duration of calls completed before this arrival; known port mean if history is empty | Input |
| waiting_hours | nonnegative hours | Berth start minus arrival | Define label only |
| is_delayed | 0 or 1 | 1 iff waiting_hours >6; equality is 0 | Target |

## Simulation assumptions

Seed: 42. Simulation begins on 1 January 2025; after warmup the exported arrivals span roughly 5-21 January. All dates and port names are fictional.

Capacities and pre-weather mean service durations are PORT_A: 3 berths /7 hours; PORT_B: 5 /8 hours; PORT_C: 7 /8.5 hours. Berths are interchangeable, operate continuously, and use first-come-first-served scheduling.

Global arrival intervals are sampled exponentially. The sampling rate is 1.35 arrivals/hour multiplied by 1.45 when the current simulation clock is between 08:00 and 18:00, otherwise by 0.72. The rate is evaluated at the beginning of each sampled interval; this is a simple time-varying approximation, not an exact nonhomogeneous Poisson simulator. Port assignment probabilities are proportional to berth counts: 3/15, 5/15, and 7/15.

Each port's weather stays constant within a 12-hour block, with severity probabilities 0.70, 0.23, and 0.07. A visit's service duration is lognormal with log-standard-deviation 0.45 and a mean corrected to the port's configured baseline. Its duration is multiplied by 1 +0.25 times the severity observed at its arrival. With probability 0.04, an unobserved disruption adds a uniform 4-16 hours. This models uncertainty; the sampled duration is hidden from model inputs.

A min-heap tracks the next free time of each berth. For each arrival: start =max(arrival, earliest free berth time); end =start +sampled service. Queue and utilization features are recorded before the arriving visit is added. Historical service duration uses only already completed calls, including warmup calls.

After labels are calculated, weather and historical-service observations are independently removed with probability 0.03 each. The saved realization has its exact missing counts in results/summary.json. Required queue and capacity counts are present and valid. Only the training partition fits imputation medians and standardization.

## Final input features

- queue_per_berth =queue_length /operational_berths
- berth_utilization =busy_berths /operational_berths
- arrival_pressure =arrivals_past_6h /operational_berths
- weather_severity
- past_avg_service_hours

The classifier does not receive visit IDs, port IDs, future operational events, elapsed final wait, labels, or future weather. Weather effects are simplified and the model's associations are not causal estimates.

## Evaluation boundaries

Nominal chronological partitions are 350 training, 50 validation, and 100 test visits. Before fitting, exclude training labels not known before the first validation arrival. Exclude validation labels not known before the first test arrival. This leaves 348 /47 /100 usable visits, with five excluded from fitting or selection. Completed later outcomes are available for retrospective test evaluation.

The model is fitted on the 348 training records. Select C from 0.1, 1, and 10 using the lowest validation Brier score. Keep the decision threshold fixed at 0.5. Neither preprocessing nor hyperparameter selection uses test outcomes. Model development should use new forward holdouts if changed after reviewing these results.

## Existing AIS CSV

data/ais_data.csv is preserved separately and is not a training source. It contains vessel descriptors and motion fields but lacks the event timestamps needed to construct the chosen target. The portable submission archive includes the generated data; the original AIS file remains in the local project.
