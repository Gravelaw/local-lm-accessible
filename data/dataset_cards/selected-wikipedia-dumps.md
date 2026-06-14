# Selected Wikipedia dumps

- source_catalog: wikimedia
- source_url: https://dumps.wikimedia.org/
- access_date: 2026-06-13
- license: CC-BY-SA-4.0
- permitted_usage: training
- commercial_use_allowed: True
- redistribution_allowed: True
- derivative_use_allowed: True
- region/country/language coverage: India, Southeast Asia, Europe, North America | India, Indonesia, France, United States | en, hi, id, fr, de
- modality: text
- task mapping: text_summarization, wikipedia_summarization
- PII/sensitive-data assessment: pii_risk=medium; contains_sensitive_data=False
- preprocessing required: verify license evidence, preserve regional metadata, large download approval and resumable checkpoint
- split usage: train, validation, test, regional_stress_test
- known limitations: requires PII review before training, metadata-only until explicitly downloaded
- approval status: approved
- reviewer notes: Selected dumps only; no full dump bundled in app runtime.
