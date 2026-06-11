# Parakeet ASR Limitations

Model: `nvidia/parakeet-tdt-0.6b-v3`.

This pipeline is local-only. Scripts read local manifests and local audio files. They do not upload audio remotely.

## Supported

Parakeet v3 is treated as supported for English and 25 European languages in this repository:

- English: `en`
- European languages tracked by the manifest gate: `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `sv`, `uk`, `ga`, `is`

## Experimental

Indian and Southeast Asian non-English ASR is marked experimental unless local eval results prove usable. This includes languages such as Hindi, Tamil, Bengali, Marathi, Telugu, Indonesian, Thai, Vietnamese, Malay, Filipino, and related regional languages.

Allowed data sources for evaluation and future experiments:

- Mozilla Common Voice
- FLEURS
- IndicVoices
- Explicit opt-in local recordings
- Synthetic noisy-room augmentation

## Required Metrics

- WER
- CER
- language detection accuracy
- unsupported-language detection
- noisy-room WER
- elderly-speaker WER
- unsupported-language failure list

## Privacy

Real recordings require license and PII metadata. Audio with personal data must be redacted or explicitly consented. Unknown licenses are rejected.

## Manifest Metadata

Every ASR manifest row must include:

- `audio_filepath`
- `duration`
- `text`
- `language`
- `region`
- `country`
- `modality`
- `task`
- `accent`
- `speaker_age_bucket`
- `license`
- `pii_status`

`modality` must be `audio`. `task` must be `speech_to_text` or `asr`.
