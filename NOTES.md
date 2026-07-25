# Notes (out-of-scope observations, not acted on)

Observations from remediation work that were out of scope for the wave in
progress at the time. Recorded here instead of being fixed inline, per the
ground rule against unscoped refactors.

## Wave 3 (resilience)

- `reporting/report_generator.py:generate_report` still `print()`s the full
  markdown report body directly to stdout, rather than through the new
  structured JSON logger introduced in this wave. This is deliberate: it's
  the run's human-facing deliverable output, not an operational trace event,
  and wrapping a multi-paragraph markdown document in a single JSON log line
  would make it unreadable from the console. Flagging in case a future wave
  wants a different convention (e.g. a dedicated "report" log level/sink).
- `api/` (the FastAPI dashboard) has no print()/logging calls at all and was
  left untouched - Wave 3.7 only covered orchestrator/evolution/eval/
  generation library code per the master prompt's explicit list.
