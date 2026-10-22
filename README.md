# Predicting port congestion and vessel waiting delays

A complete local AI/ML assignment prototype. At vessel arrival, estimate the probability of waiting **more than six hours** for a berth, return selected-outcome confidence, and explain the main factors.

**All modelling data and results are synthetic.** The existing AIS CSV is preserved separately. Everything stays local.

## Start here

- [solution.ipynb](solution.ipynb): executed simulation, training, evaluation, predictions, and checks.
- [output/pdf/port_congestion_report.pdf](output/pdf/port_congestion_report.pdf): three-page technical report.
- [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md): target, scope, and arrival-time information boundary.
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md): schema, simulation parameters, and evaluation boundaries.

## Open the notebook on this Mac

~~~bash
cd /Users/gaurav/Desktop/port-delay-prediction
source .venv/bin/activate
jupyter notebook solution.ipynb
~~~

Select **Python (port-delay-prediction)** and **Run > Run All Cells**. The notebook recreates the dataset, model, metrics, examples, and charts. Rebuild the PDF separately. Stop the server with Ctrl+C in Terminal.

## Reproduce from a fresh checkout

Use Python 3.12 (checked with 3.12.7). From the project directory:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name port-delay-prediction --display-name "Python (port-delay-prediction)"
python port_delay.py train
python -m unittest discover -s tests -v
python generate_report.py
~~~

Only the project environment is modified. The kernel lives inside it. requirements.txt pins eight direct dependencies; transitive dependencies resolve during installation. PDF generation uses Arial if available, otherwise standard Helvetica.

Check dependencies with:

~~~bash
python -m pip check
~~~

## Predict an arrival

The saved model includes preprocessing. Example inputs are the lowest/highest model probabilities in the held-out period. They illustrate behaviour and did not select model settings.

~~~bash
python port_delay.py predict --input examples/low_risk.json
python port_delay.py predict --input examples/high_risk.json
~~~

For your own input, save a JSON object such as:

~~~json
{
  "queue_length": 2,
  "operational_berths": 5,
  "busy_berths": 5,
  "arrivals_past_6h": 4,
  "weather_severity": 1,
  "past_avg_service_hours": 9.5
}
~~~

Counts must be present, finite, nonnegative integers. Capacity must be positive; occupied berths cannot exceed it. Weather is 0, 1, or 2. Weather/history may be null, with training medians used and the response flagging imputation.

The response includes delay probability, predicted outcome, confidence, main factors, signed log-odds contributions, and an advisory action. Positive contributions increase model risk relative to the training-mean reference; negative terms reduce it. These are model associations, not causal effects.

Confidence equals p for a delayed prediction and 1-p otherwise. It is not an independent reliability estimate or measured accuracy. No additional calibration model was fitted to the small validation set.

## Data and evaluation

The simulator exports 500 visits after 150 warmup visits. It models three fictional ports, variable service durations, daytime arrival peaks, persistent weather, occasional hidden disruptions, and missing measurements. Seed 42 reproduces the data. The target is 1 iff waiting hours exceed six.

Features use queue/capacity, utilization, recent arrivals/capacity, current weather, and historical completed service time. Identifiers, event outcomes, and future conditions are excluded.

Chronological nominal partitions are 350/50/100. Excluding labels unresolved at the next boundary leaves 348 training, 47 validation, and 100 test visits. Only training fits medians, scaling, and coefficients. C is selected from 0.1, 1, and 10 using validation Brier score; C=10 is selected. The alert threshold remains 0.5. Test outcomes do not tune or refit the model.

| Held-out metric (100 visits) | Prior baseline | Logistic regression |
|---|---:|---:|
| Accuracy | 87.0% | 90.0% |
| Precision | 0.0% | 66.7% |
| Recall | 0.0% | 46.2% |
| F1 | 0.000 | 0.545 |
| ROC-AUC | 0.500 | 0.770 |
| Brier score (lower better) | 0.1214 | 0.1006 |

The model catches **6 of 13 delays**, misses **7**, and raises **3 false alarms**. The baseline always predicts no delay; its precision is set to zero by convention because it makes no positive predictions.

Validation results are stronger (F1=0.833, AUC=0.993). The weaker later-period performance and few positive cases limit the conclusions. Four reliability bins contain 25 test visits each. Brier score alone does not establish calibration.

## Files and outputs

~~~text
port_delay.py                         Simulation, features, training, prediction CLI
generate_report.py                    Build report from saved results
tests/test_port_delay.py              Six checks
data/synthetic_port_calls.csv         500 generated vessel visits
examples/low_risk.json                Observed input example
examples/high_risk.json               Observed input example
results/model.joblib                 Fitted pipeline and metadata
results/summary.json                 Settings, split counts, dataset hash
results/metrics.csv                  Validation/test metrics
results/validation_candidates.csv    Model-selection evidence
results/test_predictions.csv         Held-out inputs, outcomes, predictions
results/calibration_bins.csv         Reliability summary
results/port_summary.csv             Descriptive port-level results
results/example_predictions.json     Predictions and local explanations
results/*.png                       Data, evaluation, coefficient charts
output/pdf/port_congestion_report.pdf
output/aiqon_submission.zip          Local package created after verification
~~~

The original data/ais_data.csv remains locally and lacks the timestamps needed for this target. The archive includes generated data and source files; it excludes the unused AIS file, .venv, Git metadata, and temporary files.

## Limits and operational use

High-risk alerts can prompt berth-scheduling review and downstream communication. This small synthetic study does not establish real-port accuracy, long-term seasonality, performance at unfamiliar ports, or causal benefits of interventions. Fixed capacities, interchangeable berths, and simplified weather are assumptions. Hidden remaining service times and disruptions create uncertainty.

For further development, use real event labels, select alert costs on new validation periods, and evaluate on a fresh forward holdout. Do not tune against the included test outcomes.

## Primary references

- [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Temporal evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data)
- [Python virtual environments](https://docs.python.org/3.12/library/venv.html)
