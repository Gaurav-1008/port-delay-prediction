# Local demonstration: approximately 4 minutes 20 seconds

Follow this outline live in the notebook, or use it for the assignment's optional recording. No video has been recorded.

## Prepare

Activate .venv and open solution.ipynb. It includes executed outputs. Keep the report available. Recreate results if needed with python port_delay.py train.

## 0:00-0:30 - The decision

"When a vessel arrives, estimate whether it will wait more than six hours before getting a berth."

Clarify that this predicts port waiting, the six-hour threshold is an assumption, and all modelling data are synthetic.

## 0:30-1:10 - The data

Show example rows and the port summary: one row per visit, three ports, queues, variable service, weather, and missing measurements. The later berth-start event creates the label and is excluded from inputs.

## 1:10-2:00 - Model and evaluation

Show chronological partitions and metrics. Exclude unresolved labels at cutoffs. Compare the training-prior baseline with interpretable logistic regression.

Read the actual counts: six delays detected, seven missed, three false alarms. Explain why accuracy alone is insufficient.

## 2:00-3:10 - Predictions and explanations

Run the notebook examples, or:

    python port_delay.py predict --input examples/low_risk.json
    python port_delay.py predict --input examples/high_risk.json

Show estimated probability, predicted outcome, and main factors. Queue pressure and berth occupancy push these particular scores up or down. Edit the notebook's new-arrival input to demonstrate the interface.

## 3:10-3:50 - Confidence

Confidence is the selected outcome's estimated probability, not an accuracy guarantee. Show the four-bin reliability plot. The small synthetic sample does not establish real-port reliability.

An alert could prompt berth-scheduling review and downstream notifications; the prototype does not prove intervention effects.

## 3:50-4:20 - Reproducibility and limits

Show the passed tests, saved pipeline, and report. The next step is real event records and a fresh forward holdout. Everything remains local.
