# Mozilla Common Voice

- source_catalog: huggingface
- source_url: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
- access_date: 2026-06-13
- license: CC0-1.0
- permitted_usage: training
- commercial_use_allowed: True
- redistribution_allowed: True
- derivative_use_allowed: True
- region/country/language coverage: India, Southeast Asia, Europe, North America | India, Singapore, Germany, United States | en, hi, de, fr
- modality: audio
- task mapping: speech_to_text
- PII/sensitive-data assessment: pii_risk=medium; contains_sensitive_data=False
- preprocessing required: verify license evidence, preserve regional metadata, local transcript validation, large download approval and resumable checkpoint
- split usage: train, validation, test, regional_stress_test
- known limitations: requires PII review before training, metadata-only until explicitly downloaded
- approval status: approved
- reviewer notes: Large ASR transcript dataset; no automatic download.
