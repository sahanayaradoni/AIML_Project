# ML API Monitoring Strategy

## 1. Objective

Monitor API availability, performance, prediction reliability, and model quality to detect production issues early.

## 2. Metrics to Monitor

| Metric           | What to Measure                                      | Suggested Alert                                          |
| ---------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| Availability     | Successful health checks                             | Alert after 3 consecutive failures                       |
| Error rate       | HTTP 5xx responses / total requests                  | Alert if above 5% for 5 minutes                          |
| Latency          | API response time (p95)                              | Alert if above 500 ms for 5 minutes                      |
| Request volume   | Requests per minute                                  | Investigate sudden drops or spikes                       |
| Input quality    | Missing, invalid, or out-of-range inputs             | Alert on a sustained increase                            |
| Prediction drift | Changes in input and prediction distributions        | Investigate significant deviation from baseline          |
| Model quality    | Accuracy or task-specific metrics when labels arrive | Review when performance falls below the agreed threshold |

These are initial example thresholds. They should be tuned using real production traffic and service-level objectives.

## 3. Logging and Dashboards

- Record request timestamp, endpoint, status code, and response latency.
- Record validation failures and application errors.
- Avoid logging secrets, personal information, or sensitive request payloads.
- Display availability, error rate, latency, traffic, and drift metrics on a monitoring dashboard.
- Use structured logs and centralized log storage in production.

## 4. Alerting and Response

- Notify the responsible engineer when availability or error-rate thresholds are breached.
- Investigate slow responses using latency metrics and application logs.
- Check recent deployments when errors increase suddenly.
- Roll back a faulty release when necessary.
- Document incidents, root causes, and corrective actions.

## 5. Retraining Triggers

Consider retraining when:

- Labeled production data shows sustained model-quality degradation.
- Input feature distributions show significant drift.
- Business requirements or data sources change.
- Scheduled evaluation identifies a better candidate model.

Retraining must include data validation, evaluation against the current model, approval, versioning, and rollback capability. Do not automatically deploy a new model solely because drift is detected.

## 6. Current Implementation and Limitations

This W8D1 project currently provides a baseline prediction API, a health endpoint, automated tests, and a Docker/CI build configuration.

The prediction formula is illustrative, not a trained ML model. Production monitoring, dashboards, alert delivery, drift detection, and retraining automation are strategy recommendations and are not yet implemented.
