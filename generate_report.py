"""Create the three-page technical report from saved experiment results."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
INK = colors.HexColor("#172b3a")
TEAL = colors.HexColor("#176b80")
MUTED = colors.HexColor("#516473")
LIGHT = colors.HexColor("#edf4f6")


def create_report(project=ROOT):
    project = Path(project)
    summary = json.loads((project / "results/summary.json").read_text())
    metrics = pd.read_csv(project / "results/metrics.csv")
    examples = json.loads((project / "results/example_predictions.json").read_text())
    test = metrics[metrics.split == "test"].set_index("model")
    learned, baseline = test.loc["logistic_regression"], test.loc["training_prior_baseline"]
    fonts = Path("/System/Library/Fonts/Supplemental")
    if (fonts / "Arial.ttf").is_file() and (fonts / "Arial Bold.ttf").is_file():
        pdfmetrics.registerFont(TTFont("Report", str(fonts / "Arial.ttf")))
        pdfmetrics.registerFont(TTFont("ReportBold", str(fonts / "Arial Bold.ttf")))
        pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
        normal, bold = "Report", "ReportBold"
    else:
        normal, bold = "Helvetica", "Helvetica-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyR", fontName=normal, fontSize=9.3, leading=12.7,
                              textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle("SmallR", parent=styles["BodyR"], fontSize=8.1, leading=10.7, spaceAfter=5))
    styles.add(ParagraphStyle("TitleR", fontName=bold, fontSize=24, leading=27, textColor=INK, spaceAfter=10))
    styles.add(ParagraphStyle("SectionR", fontName=bold, fontSize=11.2, leading=14,
                              textColor=TEAL, spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle("KickerR", parent=styles["SmallR"], fontName=bold,
                              textColor=TEAL, spaceAfter=8))
    styles.add(ParagraphStyle("TableR", parent=styles["SmallR"], fontSize=8.4, leading=11))
    styles.add(ParagraphStyle("TableHeadR", parent=styles["TableR"], fontName=bold, textColor=colors.white))
    def paragraph(text, style="BodyR"):
        return Paragraph(text, styles[style])
    def table(rows, widths):
        converted = [[paragraph(str(cell), "TableHeadR" if index == 0 else "TableR")
                      for cell in row] for index, row in enumerate(rows)]
        result = Table(converted, colWidths=widths, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#d8e3e7")),
        ]))
        return result
    target = project / "output/pdf/port_congestion_report.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(target), pagesize=(595.28, 841.89),
                                 leftMargin=44, rightMargin=44, topMargin=46, bottomMargin=44,
                                 title="Predicting Port Congestion and Vessel Waiting Delays",
                                 author="Gaurav", subject="Aiqon AI internship take-home assignment")
    width = 507.28
    story = [
        paragraph("AIQON / AI INTERNSHIP TAKE-HOME", "KickerR"),
        paragraph("Predicting port congestion<br/>and vessel waiting delays", "TitleR"),
        paragraph("Technical report | 21 October 2026 | Synthetic-data prototype", "SmallR"),
        paragraph("<b>Prediction:</b> at a vessel's arrival, estimate the probability that it waits "
                  "<b>more than six hours</b> before getting a berth. The output includes a selected-outcome "
                  "confidence score and an explanation. The six-hour cutoff is a prototype assumption."),
        paragraph("1. Scope and operational meaning", "SectionR"),
        paragraph("A record represents one visit to a fictional port. Arrival is entry into the designated waiting "
                  "area; berth start is the start of berth occupancy. The binary label is 1 when berth start minus "
                  "arrival exceeds six hours, and 0 otherwise, including equality. This predicts port waiting rather "
                  "than total shipment transit time. Planners could use alerts to review berth assignments and "
                  "communicate downstream disruption."),
        paragraph("2. Data and simulated ground truth", "SectionR"),
        paragraph(f"The exported data contains <b>{summary['n_records']} visits</b>, of which "
                  f"<b>{summary['delayed_visits']} ({summary['delay_rate']:.1%})</b> are delayed. The supplied local AIS "
                  "CSV lacks arrival and berthing event times, so it is preserved separately and supplies no labels. "
                  "The assignment permits synthetic data. Seed 42 and 150 warmup visits produce reproducible queue "
                  "states and historical statistics before the 500 exported arrivals."),
        table([["Fictional port", "Operational berths", "Base mean service"],
               ["PORT_A", "3", "7.0 hours"], ["PORT_B", "5", "8.0 hours"],
               ["PORT_C", "7", "8.5 hours"]], [175, 155, 177.28]),
        Spacer(1, 8),
        paragraph("Interarrival times are exponential, with an approximate sampling rate of 1.35/hour scaled by "
                  "1.45 in daytime and 0.72 otherwise. Port probabilities follow berth counts. Weather persists for "
                  "12-hour blocks (severity 0/1/2 with probabilities 0.70/0.23/0.07). Service is lognormal "
                  "(log-SD 0.45), multiplied by 1 +0.25 times arrival weather severity; 4% of visits incur an "
                  "additional hidden 4-16-hour disruption."),
        paragraph("A first-come-first-served berth heap sets start =max(arrival, earliest berth availability). "
                  "Sampled service durations and future events are hidden from model inputs. Historical averages "
                  "use only completed earlier calls, with the configured port mean as a startup fallback. After "
                  "labelling, each weather/history observation is independently masked with probability 3% "
                  f"({summary['missing_values']['weather_severity']} weather and "
                  f"{summary['missing_values']['past_avg_service_hours']} history values in this realization)."),
        paragraph("3. Features known at arrival", "SectionR"),
        paragraph("Queue per berth; fraction of berths occupied; arrivals in the previous six hours per berth; "
                  "current weather severity; and average completed service time. Normalization expresses pressure "
                  "relative to capacity. Visit/port identifiers, final wait, berth-start time, final service duration, "
                  "and future weather are excluded. Fixed capacities and interchangeable berths simplify real operations."),
        PageBreak(),
        paragraph("Evaluation and probability quality", "TitleR"),
        paragraph("4. Chronological design and model choice", "SectionR"),
        paragraph("Nominal training/validation/test sizes are 350/50/100 arrivals in time order. A short-wait label "
                  "becomes known at berth start; a positive label becomes known just after six hours of observed "
                  "waiting. Removing labels unresolved at the next partition's start leaves "
                  f"<b>{summary['split']['train']}/{summary['split']['validation']}/{summary['split']['test']}</b> "
                  "usable visits (2 training and 3 validation exclusions). Only training records fit median "
                  "imputation and standardization."),
        paragraph(f"Compare a training-prior DummyClassifier with logistic regression. Choose C from 0.1, 1, and 10 "
                  f"using the lowest validation Brier score; selected C is <b>{summary['selected_C']:g}</b>. "
                  "The classification threshold remains 0.5. No test-based tuning, oversampling, or final refit "
                  "on test observations is performed."),
        table([["Held-out metric", "Prior baseline", "Logistic regression"],
               ["Accuracy", f"{baseline.accuracy:.1%}", f"{learned.accuracy:.1%}"],
               ["Precision / recall", f"{baseline.precision:.1%} / {baseline.recall:.1%}",
                f"{learned.precision:.1%} / {learned.recall:.1%}"],
               ["F1 / ROC-AUC", f"{baseline.f1:.3f} / {baseline.roc_auc:.3f}",
                f"{learned.f1:.3f} / {learned.roc_auc:.3f}"],
               ["Brier score (lower better)", f"{baseline.brier:.4f}", f"{learned.brier:.4f}"]],
              [205, 145, 157.28]),
        Spacer(1, 10),
        Image(str(project / "results/evaluation.png"), width=width, height=width * 3.6 / 9),
        Spacer(1, 7),
        paragraph("<b>Error interpretation:</b> among 100 test visits, the model detects 6 of 13 delays, misses 7, "
                  "and raises 3 false alarms; 84 non-delays are correctly classified. Accuracy rises only three "
                  "percentage points over the no-delay baseline. The 46.2% delay recall is a material limitation "
                  "for disruption prevention. The small validation set has only seven positives; its stronger "
                  "ROC-AUC (0.993) and F1 (0.833) do not carry through to this later test period."),
        paragraph("5. What confidence means", "SectionR"),
        paragraph("The model estimates p =P(wait &gt;6h). The confidence score equals p for a delayed prediction "
                  "and 1-p for a non-delayed prediction. It is the selected outcome's model probability, not an "
                  "independent reliability score or measured accuracy. Four quantile reliability bins compare "
                  "estimated probabilities and observed fractions. Brier score combines several properties and "
                  "does not establish calibration alone. No additional calibrator is fitted to the small validation "
                  "sample. Extreme probabilities and real-port reliability remain unverified."),
        PageBreak(),
        paragraph("Explanations and operational use", "TitleR"),
        paragraph("6. Exact model explanations", "SectionR"),
        paragraph("For each standardized input z, contribution =coefficient times z. Contributions plus the "
                  "intercept reconstruct the model's log-odds; applying the logistic function returns the prediction. "
                  "A positive term raises delay risk relative to the training-mean reference, and a negative term "
                  "lowers it. These are associations within this model, not causal effects or percentage-point changes."),
    ]
    example_rows = [["Held-out illustration", "Low predicted risk", "High predicted risk"]]
    low, high = examples["low_risk"], examples["high_risk"]
    example_rows += [
        ["Visit / outcome", f"{low['source_visit']} / not delayed", f"{high['source_visit']} / delayed"],
        ["Estimated delay probability", f"{low['result']['delay_probability']:.5%}",
         f"{high['result']['delay_probability']:.2%}"],
        ["Queue / occupied / capacity",
         f"{low['observation']['queue_length']:g} / {low['observation']['busy_berths']:g} / {low['observation']['operational_berths']:g}",
         f"{high['observation']['queue_length']:g} / {high['observation']['busy_berths']:g} / {high['observation']['operational_berths']:g}"],
    ]
    story += [
        table(example_rows, [195, 156, 156.28]),
        Spacer(1, 8),
        paragraph("These illustrations were selected as the lowest and highest test probabilities after evaluation; "
                  "they did not select model settings and are not evidence of typical prediction quality. Queue "
                  "pressure and berth occupancy dominate these particular explanations. Missing features are "
                  "explicitly flagged when filled with training medians."),
        Image(str(project / "results/feature_coefficients.png"), width=460, height=460 * 3.2 / 7),
        paragraph("7. Actions, limitations, and next steps", "SectionR"),
        paragraph("<b>Use:</b> a high-risk alert prompts review of berth scheduling and notifications to downstream "
                  "planners. A low-risk score supports continued monitoring. Outputs are advisory; the model has not "
                  "demonstrated that any intervention will reduce delay."),
        paragraph("<b>Limits:</b> synthetic labels validate the simulator-to-model workflow. The short date range, "
                  "three seen ports, fixed capacities, simplified weather, and small number of positive outcomes "
                  "do not establish real-port accuracy, seasonal effects, generalization to new ports, or dependable "
                  "probabilities. Hidden remaining service times and disruptions create irreducible uncertainty "
                  "for the chosen inputs."),
        paragraph("<b>Next:</b> collect real arrival/berth events and live queue snapshots. Validate at later time "
                  "periods and unseen ports. Choose alert thresholds against explicit missed-delay and false-alarm "
                  "costs using new validation periods, then evaluate on a fresh forward holdout."),
        paragraph("Reproduction and references", "SectionR"),
        paragraph("Run python port_delay.py train; python -m unittest discover -s tests -v; "
                  "python generate_report.py. The executed notebook, generated CSV, fitted pipeline, metrics, charts, "
                  "example JSON inputs, and demo outline are included locally. Requirements pin the direct packages.",
                  "SmallR"),
        paragraph('Primary documentation: '
                  '<link href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" color="#176b80">scikit-learn logistic regression</link>; '
                  '<link href="https://scikit-learn.org/stable/modules/calibration.html" color="#176b80">probability calibration</link>; '
                  '<link href="https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data" color="#176b80">temporal evaluation</link>. '
                  'Assignment brief: Aiqon AI Internship Take-Home Assignment (3 Days).', "SmallR"),
    ]
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5e2e7"))
        canvas.line(44, 34, 551, 34)
        canvas.setFont(normal, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(44, 21, "AIQON / SYNTHETIC PORT-DELAY STUDY / SEED 42")
        canvas.drawRightString(551, 21, f"{doc.page} / 3")
        canvas.restoreState()
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(target)
    return target


if __name__ == "__main__":
    create_report()
