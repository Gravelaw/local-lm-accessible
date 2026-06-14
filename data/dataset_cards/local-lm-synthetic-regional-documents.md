# local-lm synthetic regional documents

- source_catalog: synthetic
- source_url: file:data/synthetic
- access_date: 2026-06-13
- license: Apache-2.0
- permitted_usage: training
- commercial_use_allowed: True
- redistribution_allowed: True
- derivative_use_allowed: True
- region/country/language coverage: India, Southeast Asia, Europe, North America | India, Singapore, France, United States | en, hi, fr
- modality: document_image
- task mapping: bank_statement_extraction, bill_extraction, document_ocr, invoice_extraction, receipt_extraction
- PII/sensitive-data assessment: pii_risk=none; contains_sensitive_data=False
- preprocessing required: verify license evidence, preserve regional metadata, local OCR/image preprocessing
- split usage: train, validation, test, regional_stress_test
- known limitations: none recorded
- approval status: approved
- reviewer notes: Repo-generated synthetic data; safe for training after generation.
