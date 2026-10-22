# Prediction problem definition

Status: implemented locally. See README.md, DATA_DICTIONARY.md, the executed notebook, and the technical report for the completed experiment.

## Objective

At the moment a vessel arrives at a port's designated waiting area, estimate the probability that it will wait more than six hours before starting its berth visit. Return the predicted outcome, its estimated probability, and the main model factors behind that estimate.

This treats excessive waiting before berthing as the project's indicator of congestion-related delay. It does not estimate total shipment transit time or delay relative to a contractual delivery deadline.

## Assignment requirements and selected assumptions

The assignment requires a congestion/delay likelihood, a confidence score, explanations, and an evaluation. It permits synthetic data and leaves the precise congestion definition to the candidate.

The following are our project choices:

- One record represents one vessel visit to one port. A vessel can have multiple visits, so its vessel identifier alone is not a unique record key.
- Prediction happens at arrival, before the vessel's waiting outcome is known.
- Arrival means entry into the designated port waiting area; berth start means the start of berth occupancy. These event definitions must be applied consistently.
- A waiting time strictly greater than six hours is considered delayed. Six hours is a prototype assumption, not an industry standard or a threshold supplied by the assignment.
- Timestamps will use a consistent timezone, preferably UTC, and durations will be expressed in hours.

## Target and ground truth

For a completed observation:

```python
waiting_hours = (berth_start_time - arrival_time).total_seconds() / 3600
is_delayed = int(waiting_hours > 6)
```

The model estimates:

```text
P(waiting_hours > 6 | information available at arrival)
```

`waiting_hours` and `is_delayed` are used for labelling and evaluation, never as model inputs.

| Arrival | Berth start | Waiting hours | is_delayed |
|---|---|---:|---:|
| 08:00 | 12:00, same day | 4 | 0 |
| 08:00 | 14:00, same day | 6 | 0 |
| 08:00 | 16:00, same day | 8 | 1 |

Reject or investigate records where berth start precedes arrival. A vessel still waiting when observation ends must not automatically be labelled 0. Its positive label is already known if it has been observed waiting for more than six hours; otherwise its outcome remains unresolved and it is excluded from binary supervised training until resolved. At a training cutoff, use only labels that would have been known by that cutoff.

## Information available at prediction time

Candidate inputs are:

| Input | Definition at arrival |
|---|---|
| Queue length | Other vessels already waiting, excluding the arriving vessel |
| Operational berths | Berths currently in service |
| Busy berths | Operational berths currently occupied |
| Recent arrivals | Vessel arrivals in the preceding six hours |
| Weather severity | Conditions observed at arrival, or a forecast issued by then |
| Historical service time | Average service duration from visits completed before this arrival |

Two derived features are `queue_length / operational_berths` and `busy_berths / operational_berths`. The initial prototype assumes at least one operational berth. Historical features need a documented fallback when no previous completed visits exist.

Actual future weather, eventual berth start, final service duration for this visit, and statistics calculated from future records are excluded from the input features.

## Dataset

The implementation uses 500 synthetic vessel visits with recorded arrival conditions and simulated berth-start times. DATA_DICTIONARY.md documents the simulator, and results/summary.json records the realized dataset. The metrics describe performance in that simulated environment.

If real operational data becomes available, it must provide consistently defined arrival and berth-start events for each port visit, together with input measurements available at arrival.

## Output contract

For each valid arrival, return:

- `delay_probability`: the model's estimated probability of a wait greater than six hours.
- `prediction`: delayed or not delayed, using a decision threshold initially set to 0.5; any later adjustment uses validation data.
- `confidence_score`: the probability assigned to the predicted class: `p` for delayed and `1 - p` for not delayed. This is not a separate reliability estimate or overall model accuracy.
- `main_factors`: the strongest factors in this specific model prediction, translated into plain language.

Probability reliability must be assessed during evaluation. Explanations describe model behaviour rather than proving causal effects. Measured results and prediction examples are available in results/ and the technical report.

## Operational use and completion criteria

A high predicted risk can prompt planners to review berth scheduling and communicate potential disruption to downstream teams. It does not guarantee that a suggested operational action will reduce the delay.

This document specifies the prediction event, six-hour target, label calculation, arrival-time information boundary, dataset limitations, and required outputs. The completed implementation follows these definitions; its measured performance and limitations are documented separately.
