# Business Requirements Document  
## Project: **local-lm — Backyard AI Accessibility Assistant**

**Version:** 1.0  
**Hackathon:** Hugging Face Build Small Hackathon  
**Track:** Backyard AI  
**Target users:** Elderly users, users with low vision, users with accessibility issues, and non-technical users who need local AI assistance for documents, images, voice, and simple knowledge tasks.

---

## 1. Executive Summary

`local-lm` is a local-first, small-model assistant designed for elderly and accessibility-constrained users. The system will help users read, understand, convert, summarize, and translate information from documents, photos, web/Wikipedia pages, and speech.

The app will be implemented as a **Gradio application hosted on Hugging Face Spaces**, because the hackathon requires the submission to be a Gradio app hosted as a Hugging Face Space. The model stack must remain under **32B total parameters**, and the Backyard AI track rewards specific, real-world problem solving, honest use of the small-model constraint, and Gradio polish.

The product vision is:

```text
A simple, local-first AI assistant that helps elderly or low-vision users read the world around them:
documents, bills, signs, handwritten notes, web pages, and spoken input.
```

The core design decision is to avoid one large generalist model. Instead, `local-lm` will use a **small routed model bundle**:

```text
Text model      → summarization, routing, explanations, tool planning
Vision model    → OCR, document extraction, photo description, image translation
ASR model       → speech-to-text
Python tools    → PDF, Excel, TXT, JSON, image preprocessing, local search
Gradio app      → accessible UI
```

---

## 2. Hackathon Context and Strategic Fit

### 2.1 Hackathon Constraints

| Constraint | Requirement |
|---|---|
| Model size | Total parameters must be ≤ 32B |
| Runtime expectation | Model should fit on a laptop |
| App framework | App must be built on Gradio |
| Hosting | App must be hosted as a Hugging Face Space |
| Submission assets | Space link, short demo video, and social post |
| Relevant track | Backyard AI |

### 2.2 Bonus Quests Targeted

`local-lm` should deliberately target the following bonus quests:

| Bonus quest | How local-lm targets it |
|---|---|
| **Off the Grid** | No cloud inference, no cloud OCR, no remote telemetry |
| **Local-first** | Models run locally inside the Space/container and laptop-local mode is planned |
| **Well-Tuned** | Fine-tuned LoRA adapters for document extraction, summarization, routing, and accessibility style |
| **Llama Champion** | Use `llama.cpp` / GGUF where possible |
| **Off-Brand** | Custom accessible Gradio UI beyond default components |
| **Field Notes** | Publish a technical write-up/report |
| **Best Agent** | Local deterministic router plus model-assisted tool planning |

---

## 3. Problem Statement

Elderly users and users with low vision often struggle with:

```text
- reading small printed text
- understanding bills and bank statements
- converting paper documents into usable digital formats
- summarizing long articles or Wikipedia pages
- translating signs, labels, menus, or notices
- interacting with software using typing-heavy interfaces
- trusting AI systems that send private documents to cloud services
```

Most LLM apps assume the user can type, read dense output, understand model limitations, and manage cloud privacy tradeoffs. `local-lm` addresses this by providing a **task-first, accessibility-first, local-first interface**.

The product should prioritize:

```text
simple language
large controls
voice input
image upload/capture
read-aloud-ready responses
clear warnings
structured exports
no hidden cloud inference
```

---

## 4. Business Objectives

### 4.1 Primary Objectives

| Objective | Description |
|---|---|
| Improve accessibility | Help elderly and low-vision users understand documents, images, and web content |
| Preserve privacy | Keep inference local; do not upload user files to cloud APIs |
| Demonstrate small-model usefulness | Stay under 32B parameters while solving practical tasks |
| Deliver a polished hackathon demo | Provide a working Gradio Space with demo video and reproducible repo |
| Support regional usefulness | Focus data and evaluation on India, Southeast Asia, North America, and Europe |
| Enable future laptop-local deployment | Design the Space architecture so it can later run on a user’s laptop |

### 4.2 Hackathon Success Objective

Win or place competitively in:

```text
- Backyard AI main track
- Llama Champion
- Well-Tuned
- Off the Grid
- NVIDIA Nemotron Quest
- OpenBMB special category
- Best Agent
- Best Demo
```

---

## 5. Product Scope

### 5.1 In Scope for Hackathon MVP

The hackathon MVP should include:

```text
1. Gradio app hosted as a Hugging Face Space
2. Text summarization:
   - Wikipedia text
   - pasted article text
   - uploaded text/PDF documents
   - optional URL fetch if allow_web=true

3. Document conversion:
   - invoice/bill/receipt image or PDF → JSON
   - invoice/bill/receipt image or PDF → Excel
   - handwritten note image → TXT
   - bank statement image/PDF → structured transaction table, with human-review warning

4. Image accessibility:
   - describe image in simple language
   - detect visible text
   - highlight possible hazards
   - generate speech-friendly response

5. Image translation:
   - extract visible text
   - translate to selected language
   - preserve original and translated text

6. Speech-to-text:
   - basic audio upload / microphone transcription
   - English and supported European languages first

7. Local orchestration:
   - deterministic router
   - model clients
   - Python tool execution
   - export to TXT, JSON, XLSX, PDF

8. Fine-tuning artifacts:
   - LoRA/QLoRA training pipeline
   - dataset registry
   - synthetic document generator
   - local eval harness
```

### 5.2 Out of Scope for Hackathon MVP

```text
- Full mobile native app
- Full offline Wikipedia dump shipped inside the Space
- Full Indian/SEA speech-to-text support
- Medical diagnosis
- Legal advice
- Financial advice
- Real bank-statement training without explicit consent/redaction
- Cloud OCR
- Cloud LLM APIs
- Full pretraining
- Multi-user account system
```

---

## 6. User Personas

### 6.1 Primary Persona: Elderly User with Low Technical Literacy

| Attribute | Description |
|---|---|
| Needs | Read documents, understand bills, describe images |
| Pain points | Small text, confusing UI, privacy concerns |
| UX requirement | Large buttons, clear outputs, minimal jargon |
| Success outcome | User uploads/captures an image and receives a simple, useful answer |

### 6.2 Secondary Persona: Low-Vision User

| Attribute | Description |
|---|---|
| Needs | Image description, OCR, visible text reading, translation |
| Pain points | Poor contrast, inaccessible PDFs, images with embedded text |
| UX requirement | High contrast, speech-ready response, simple navigation |
| Success outcome | User understands an image, sign, label, or document without assistance |

### 6.3 Tertiary Persona: Caregiver or Family Member

| Attribute | Description |
|---|---|
| Needs | Help elderly person process documents safely |
| Pain points | Time burden, repetitive document reading |
| UX requirement | Review mode, confidence flags, export options |
| Success outcome | Converts bills/statements into readable tables with warnings |

### 6.4 Developer / Hackathon Judge

| Attribute | Description |
|---|---|
| Needs | Verify constraints, model size, local-first design, reproducibility |
| Pain points | Hidden cloud dependencies, demo-only mockups |
| UX requirement | Clear README, model manifest, local smoke tests |
| Success outcome | Can run/evaluate the app and verify model budget |

---

## 7. Key Use Cases

### UC-01: Summarize Wikipedia or Article Text

**User story:**  
As an elderly user, I want to paste a Wikipedia/article text or provide a URL so I can receive a short, simple summary.

**Inputs:**

```text
- pasted text
- uploaded .txt/.pdf
- optional URL
```

**Processing:**

```text
1. Router detects summarization task.
2. Text is cleaned and chunked.
3. Text model summarizes each chunk.
4. Text model produces final simple-language summary.
5. Safety checker prevents unsupported factual claims.
```

**Outputs:**

```text
- short summary
- key points
- optional detailed summary
- source/chunk references where available
```

**Acceptance criteria:**

```text
- summary uses simple language
- no unsupported claims
- output generated locally
- web fetching disabled unless allow_web=true
```

---

### UC-02: Convert Invoice/Bill/Receipt to JSON and Excel

**User story:**  
As a user, I want to upload a bill or receipt photo and get a structured Excel file.

**Inputs:**

```text
- image file
- PDF file
- document type: auto/invoice/bill/receipt
```

**Processing:**

```text
1. Image preprocessing: deskew, contrast, orientation correction.
2. Vision model extracts document text and fields.
3. Output is forced into a Pydantic schema.
4. Numeric validation checks subtotal, tax, and total.
5. Excel export tool writes .xlsx.
6. Warnings are shown for low confidence or inconsistent totals.
```

**Outputs:**

```text
- JSON
- Excel file
- TXT OCR output
- optional PDF report
- warnings and confidence
```

**Acceptance criteria:**

```text
- valid JSON ≥99% on eval set
- no silent financial total hallucination
- low-confidence extraction requires user review
- Excel file opens successfully
```

---

### UC-03: Convert Bank Statement to Transaction Table

**User story:**  
As a caregiver, I want to convert a bank statement image/PDF into a transaction table while clearly seeing uncertain rows.

**Inputs:**

```text
- bank statement image/PDF
```

**Processing:**

```text
1. Image/PDF preprocessing.
2. Vision model extracts table rows.
3. Schema validator checks dates, debits, credits, balances.
4. Reconciliation validator checks running balances where possible.
5. Output is flagged as review-required.
```

**Outputs:**

```text
- transaction JSON
- Excel transaction table
- warnings
- review-required flag
```

**Acceptance criteria:**

```text
- every bank-statement export includes review warning
- no final financial interpretation is presented as authoritative
- uncertain rows are highlighted
```

---

### UC-04: Describe Image for Accessibility

**User story:**  
As a low-vision user, I want to upload/capture a photo and receive a simple description.

**Inputs:**

```text
- image file
```

**Processing:**

```text
1. Router detects image description.
2. Vision model describes scene.
3. Safety prompt prioritizes hazards and visible text.
4. Response is simplified for speech/read-aloud.
```

**Outputs:**

```json
{
  "short_description": "",
  "visible_text": [],
  "possible_hazards": [],
  "uncertainties": [],
  "spoken_response": ""
}
```

**Acceptance criteria:**

```text
- hazards mentioned first when visible
- visible text separated from general description
- no identity guessing
- uncertainty stated when ambiguous
```

---

### UC-05: Translate Visible Text in Image

**User story:**  
As a user, I want to upload an image of a sign, label, menu, or document and translate the visible text.

**Inputs:**

```text
- image
- target language
```

**Processing:**

```text
1. Vision model extracts text.
2. Language is detected or user-selected.
3. Text model or vision model translates.
4. Output keeps original and translation.
```

**Outputs:**

```text
- original visible text
- translated text
- confidence/warnings
```

**Acceptance criteria:**

```text
- original OCR is shown
- translation is shown separately
- uncertain text is marked
```

---

### UC-06: Speech-to-Text

**User story:**  
As a user with typing difficulty, I want to speak and get text input into the assistant.

**Inputs:**

```text
- microphone audio
- uploaded .wav/.flac
```

**Processing:**

```text
1. Audio is captured/uploaded.
2. ASR model transcribes.
3. Transcript can be sent to router.
```

**Outputs:**

```text
- transcript
- detected language where supported
- confidence/limitations
```

**Acceptance criteria:**

```text
- audio remains local
- unsupported languages are not falsely claimed as supported
- Indian/SEA non-English ASR marked experimental unless evaluated
```

---

## 8. Functional Requirements

### 8.1 Gradio Application

| ID | Requirement | Priority |
|---|---|---|
| FR-001 | App shall be implemented in Gradio | Must |
| FR-002 | App shall be hosted as a Hugging Face Space | Must |
| FR-003 | App shall provide large-button task modes | Must |
| FR-004 | App shall provide upload controls for image, PDF, text, and audio | Must |
| FR-005 | App shall provide downloadable JSON, TXT, XLSX, and PDF outputs | Must |
| FR-006 | App shall include an accessibility-oriented UI mode | Must |
| FR-007 | App shall show model/runtime/locality status | Should |
| FR-008 | App shall include sample files for demo | Should |
| FR-009 | App shall support custom CSS for high contrast and large fonts | Should |

### 8.2 Task Router

| ID | Requirement | Priority |
|---|---|---|
| FR-010 | System shall use deterministic routing first | Must |
| FR-011 | Router shall select summarization, document extraction, image description, image translation, or ASR | Must |
| FR-012 | Router shall avoid calling multiple models unnecessarily | Must |
| FR-013 | Router shall provide a human-readable reason for selected task | Should |
| FR-014 | Router shall allow manual override | Should |

### 8.3 Text Model Service

| ID | Requirement | Priority |
|---|---|---|
| FR-020 | Text model shall summarize articles/documents | Must |
| FR-021 | Text model shall produce simple-language summaries | Must |
| FR-022 | Text model shall assist with JSON repair | Should |
| FR-023 | Text model shall produce tool-call plans in strict JSON | Should |
| FR-024 | Text model shall include uncertainty warnings for sensitive tasks | Must |

The selected text model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` for deployment. It is used for text summarization, routing, tool-call planning, and elderly-friendly explanations.

### 8.4 Vision/OCR Service

| ID | Requirement | Priority |
|---|---|---|
| FR-030 | Vision model shall extract text from document images | Must |
| FR-031 | Vision model shall output invoice/receipt/bill JSON | Must |
| FR-032 | Vision model shall extract bank-statement transactions | Must |
| FR-033 | Vision model shall transcribe handwritten notes | Should |
| FR-034 | Vision model shall describe photos for accessibility | Must |
| FR-035 | Vision model shall translate visible text | Should |
| FR-036 | Vision model shall return uncertainty flags | Must |

The selected vision model is `openbmb/MiniCPM-V-4.6`.

### 8.5 Speech-to-Text Service

| ID | Requirement | Priority |
|---|---|---|
| FR-040 | System shall transcribe uploaded audio | Must |
| FR-041 | System shall support microphone input in Gradio where available | Should |
| FR-042 | System shall expose unsupported-language warnings | Must |
| FR-043 | System shall mark Indian/SEA non-English ASR as experimental | Must |
| FR-044 | System shall allow transcript handoff to task router | Should |

### 8.6 File Export

| ID | Requirement | Priority |
|---|---|---|
| FR-050 | System shall export OCR/extraction output to JSON | Must |
| FR-051 | System shall export tables to XLSX | Must |
| FR-052 | System shall export raw OCR text to TXT | Must |
| FR-053 | System shall generate PDF summaries/reports | Should |
| FR-054 | System shall include confidence and warnings in exports | Must |

### 8.7 Dataset Registry and Training Pipeline

| ID | Requirement | Priority |
|---|---|---|
| FR-060 | System shall maintain dataset registry | Must |
| FR-061 | Every dataset shall have license metadata | Must |
| FR-062 | Every dataset shall have region/language/task metadata | Must |
| FR-063 | Unknown-license datasets shall be rejected | Must |
| FR-064 | High-PII datasets shall be blocked unless synthetic/redacted/opt-in | Must |
| FR-065 | Synthetic regional document generation shall be supported | Must |
| FR-066 | LoRA/QLoRA text fine-tuning shall be supported | Must |
| FR-067 | MiniCPM-V multimodal fine-tuning shall be supported | Must |
| FR-068 | Parakeet ASR fine-tuning shall be eval-first/experimental | Should |

---

## 9. Non-Functional Requirements

### 9.1 Privacy and Locality

| ID | Requirement | Priority |
|---|---|---|
| NFR-001 | No cloud inference in shipped product | Must |
| NFR-002 | No cloud OCR | Must |
| NFR-003 | No remote telemetry by default | Must |
| NFR-004 | User files shall not be uploaded to third-party APIs | Must |
| NFR-005 | Web fetch shall be disabled by default | Must |
| NFR-006 | App shall disclose when Hugging Face Space compute is being used | Must |
| NFR-007 | Laptop-local mode shall be documented | Should |

Important nuance: a Hugging Face Space is hosted on Hugging Face infrastructure, not literally on the user’s laptop. For hackathon compliance, the demo must be a Gradio Space. For the “local-first” claim, the correct interpretation is:

```text
No external model APIs or cloud OCR are called.
All inference runs inside the app runtime using downloaded models.
A laptop-local mode is documented and supported by the same architecture.
```

### 9.2 Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-010 | Text summary first response | ≤ 20 seconds for short input |
| NFR-011 | Image description | ≤ 30 seconds for single image |
| NFR-012 | Invoice extraction | ≤ 60 seconds for single page |
| NFR-013 | Audio transcription | ≤ 1.5x audio duration for short clips |
| NFR-014 | Space cold start | Best effort; documented |
| NFR-015 | Laptop mode memory | Target 16–32GB RAM with quantized models |

### 9.3 Usability and Accessibility

| ID | Requirement | Priority |
|---|---|---|
| NFR-020 | Large text and large buttons | Must |
| NFR-021 | High-contrast mode | Must |
| NFR-022 | Minimal technical terminology | Must |
| NFR-023 | One-task-per-panel layout | Must |
| NFR-024 | Speech-ready output | Should |
| NFR-025 | Clear confidence/warning messages | Must |
| NFR-026 | No model names shown to end user by default | Should |

### 9.4 Safety

| ID | Requirement | Priority |
|---|---|---|
| NFR-030 | Medical/legal/financial outputs must include review warnings | Must |
| NFR-031 | Bank statements must always require human review | Must |
| NFR-032 | Medication labels may be read but dosage advice must not be given | Must |
| NFR-033 | Image descriptions must not guess identity | Must |
| NFR-034 | Ambiguous visual content must include uncertainty | Must |
| NFR-035 | Low-confidence OCR must not be silently exported as final truth | Must |

---

## 10. Model Requirements

### 10.1 Model Budget

| Model | Purpose | Parameters |
|---|---|---:|
| NVIDIA Nemotron-3-Nano-4B | Text, routing, summarization, tool planning | 3.97B |
| OpenBMB MiniCPM-V-4.6 | Vision, OCR, document extraction, image description | Small VLM class; deployable with quantized variants |
| NVIDIA Parakeet TDT 0.6B v3 | Speech-to-text | 0.6B |
| Optional MiniCPM-o-4.5 | Future multimodal/omni mode | 9B |

Minimum v1 parameter load is approximately:

```text
Nemotron 3.97B + Parakeet 0.6B + MiniCPM-V ≈ far below 32B
```

The project remains safely within the hackathon’s ≤32B parameter cap.

### 10.2 Model Selection Rationale

| Model | Rationale |
|---|---|
| Nemotron-3-Nano-4B | Small, edge-oriented, GGUF available, `llama.cpp` support, commercial-ready |
| MiniCPM-V-4.6 | Strong fit for document OCR, mobile/edge, GGUF/llama.cpp ecosystem, fine-tuning support |
| Parakeet TDT 0.6B v3 | Small ASR model, supports English and European languages |
| BFL models | Not v1 priority because image generation/editing is not core to the use case |
| Cohere models | Possible future comparison, but licensing and parameter fit must be reviewed per model |

---

## 11. Training Requirements

### 11.1 Fine-Tuning Method

The project shall use:

```text
- LoRA / QLoRA
- not full fine-tuning
- not pretraining
```

Training stack:

```text
Text model:
- Transformers
- TRL
- PEFT
- bitsandbytes
- accelerate

Vision model:
- LLaMA-Factory preferred
- SWIFT fallback

ASR:
- NVIDIA NeMo
- Transformers
- eval-first
```

### 11.2 Text Model Training Objectives

Nemotron shall be fine-tuned for:

```text
- task routing
- summarization
- simple elderly-friendly explanations
- local tool-call planning
- JSON repair
- uncertainty and safety warnings
```

Example target output:

```json
{
  "route": "document_to_excel",
  "tool": "extract_invoice",
  "arguments": {
    "file_path": "<local_path>",
    "document_type": "invoice",
    "output_formats": ["json", "xlsx", "txt"]
  },
  "needs_user_confirmation": true,
  "reason": "The user uploaded a bill image and requested Excel conversion."
}
```

### 11.3 Vision Model Training Objectives

MiniCPM-V shall be fine-tuned for:

```text
- image/PDF OCR
- invoice extraction
- receipt extraction
- bill extraction
- bank-statement table extraction
- handwritten note transcription
- accessibility image description
- image text translation
```

### 11.4 ASR Training/Eval Objectives

Parakeet shall be used initially for:

```text
- English ASR
- supported European-language ASR
- Indian English evaluation
- noisy-room evaluation
- elderly-speaker evaluation
```

Indian and Southeast Asian non-English ASR shall be considered experimental until metrics support it.

---

## 12. Data Requirements

### 12.1 Regional Focus

Training and evaluation data shall focus on:

```text
- India
- Southeast Asia
- North America
- Europe
```

Default training mix:

| Region | General tasks |
|---|---:|
| India | 35% |
| Southeast Asia | 20% |
| North America | 25% |
| Europe | 20% |

Document extraction mix:

| Region | Document tasks |
|---|---:|
| India | 40% |
| Southeast Asia | 25% |
| North America | 20% |
| Europe | 15% |

### 12.2 Dataset Discovery Sources

The dataset registry shall support discovery from:

```text
- Awesome Public Datasets
- AWS Registry of Open Data
- Azure Open Datasets
- UCI ML Repository
- Google Dataset Search
- Google Cloud Marketplace free datasets
- Wikimedia
- Hugging Face datasets
- synthetic data
- manual manifests
- opt-in redacted user samples
```

These are discovery sources only. No dataset may enter training until accepted by the dataset gate.

### 12.3 Dataset Acceptance Rules

```text
1. Unknown license → reject
2. Non-commercial license → research/eval only unless explicitly allowed
3. High PII risk → block unless synthetic/redacted/opt-in
4. Financial/medical/legal/identity documents → block by default
5. Dataset must have region, country, language, modality, and task tags
6. Dataset must have a dataset card
7. Dataset must be stored locally before training
```

### 12.4 Synthetic Data Requirements

Synthetic document generation is mandatory for:

```text
- invoices
- receipts
- utility bills
- bank statements
- handwritten notes
- multilingual signs/labels
```

Regional synthetic coverage:

```text
India:
- GSTIN-like synthetic IDs
- HSN/SAC
- CGST/SGST/IGST
- UPI references
- INR formatting
- Hindi/Tamil/Bengali/Marathi/Telugu labels

Southeast Asia:
- Singapore GST
- Malaysia SST
- Indonesia PPN/VAT
- Thailand VAT
- Philippines VAT
- Vietnam VAT
- local currencies/date formats

North America:
- US sales tax
- Canadian GST/HST/PST
- USD/CAD formatting
- merchant receipts
- masked card digits

Europe:
- VAT invoices
- EUR/GBP/CHF formatting
- VAT-ID-like synthetic fields
- IBAN-like synthetic fields
- decimal comma formats
- French/German/Spanish/Italian/Dutch/Portuguese labels
```

---

## 13. System Architecture

### 13.1 Logical Architecture

```text
Gradio UI
  ↓
Local task router
  ↓
Tool/model orchestration layer
  ├── Text model client
  │     └── llama.cpp / GGUF Nemotron
  ├── Vision model client
  │     └── MiniCPM-V via llama.cpp or Transformers
  ├── ASR service
  │     └── Parakeet via NeMo/Transformers
  ├── Python tools
  │     ├── image preprocessing
  │     ├── PDF parsing
  │     ├── Excel export
  │     ├── TXT/JSON/PDF export
  │     ├── local Wikipedia/document search
  │     └── safety validation
  └── Evaluation and logging layer
```

### 13.2 Deployment Architecture

```text
Hugging Face Space
  ├── Gradio app
  ├── local model files or downloaded model cache
  ├── llama.cpp server for text model where possible
  ├── local vision runtime
  ├── local ASR runtime
  ├── local file-processing tools
  └── no external inference APIs
```

### 13.3 Laptop-Local Architecture

```text
Laptop
  ├── app.py / Gradio
  ├── llama-server: text model
  ├── vision model service
  ├── ASR service
  ├── local model cache
  ├── local SQLite index
  └── local exports folder
```

---

## 14. Gradio UX Requirements

### 14.1 UI Layout

The app shall use a task-first interface:

```text
[Read / Summarize]
[Convert Document]
[Describe Image]
[Translate Image Text]
[Speech to Text]
[Settings / Privacy]
```

### 14.2 Accessibility Design

| UI element | Requirement |
|---|---|
| Buttons | Large, high-contrast |
| Text | Large default font |
| Outputs | Short answer first, details expandable |
| Warnings | Clear and visible |
| Downloads | One-click |
| Navigation | Minimal tabs |
| Model details | Hidden by default, visible in advanced panel |
| Privacy status | Always visible |

### 14.3 Output Design

Every task output should show:

```text
1. Simple answer
2. Confidence
3. Warnings
4. Extracted text or structured result
5. Download buttons
6. “Needs human review” flag when relevant
```

---

## 15. Evaluation Requirements

### 15.1 Text Model Metrics

| Metric | Target |
|---|---:|
| Router accuracy | ≥95% |
| Tool-call JSON validity | ≥99% |
| Summary factual coverage | ≥90% on curated eval |
| Unsafe certainty | 0 critical failures |
| Elderly readability | Simple-language output |

### 15.2 Document Extraction Metrics

| Metric | Target |
|---|---:|
| JSON schema validity | ≥99% |
| Invoice field F1 | ≥90% |
| Receipt field F1 | ≥90% |
| Bank transaction row F1 | ≥85% initial target |
| Numeric consistency | ≥98% |
| Low-confidence flag recall | ≥95% |
| Silent wrong financial total | 0 critical failures |

### 15.3 Image Accessibility Metrics

| Metric | Target |
|---|---:|
| Hazard-first response when hazard visible | ≥95% |
| Visible text recall | ≥90% |
| Identity guessing | 0 critical failures |
| Uncertainty stated when ambiguous | ≥90% |
| Speech-friendly response length | <80 words default |

### 15.4 ASR Metrics

| Metric | Target |
|---|---:|
| English WER | Baseline tracked |
| Indian English WER | Tracked separately |
| Elderly speech WER | Tracked separately |
| Noisy-room WER | Tracked separately |
| Unsupported-language hallucination | 0 critical failures |

### 15.5 Hackathon Demo Acceptance Criteria

```text
- Space loads
- Gradio UI is usable
- At least one sample invoice converts to Excel
- At least one image is described accessibly
- At least one article is summarized
- At least one audio sample transcribes
- Model budget is shown
- No cloud API keys are required
- README explains local-first behavior and limitations
```

---

## 16. Technical Deliverables

### 16.1 Repository Structure

```text
local-lm/
  README.md
  AGENTS.md
  app.py
  requirements.txt
  pyproject.toml

  local_lm/
    ui/
      gradio_app.py
      components.py
      styles.css
    routing/
      router.py
      intents.py
    models/
      text_client.py
      vision_client.py
      asr_client.py
      manifest.py
    tools/
      web_fetch.py
      wiki_index.py
      pdf_extract.py
      image_preprocess.py
      excel_export.py
      pdf_export.py
      safety_checks.py
    schemas/
      document_schemas.py
      source_registry.py
      outputs.py

  training/
    text/
      train_nemotron_lora.py
      merge_adapter.py
      export_gguf.sh
      quantize_gguf.sh
    vision/
      convert_to_llamafactory.py
      train_minicpmv_lora.sh
      eval_minicpmv.py
    asr/
      prepare_manifest.py
      eval_wer.py
      train_parakeet_nemo.sh

  data/
    registry/
    synthetic/
    processed/
    splits/

  evals/
    run_all_evals.py
    text_eval.py
    vision_eval.py
    asr_eval.py
    report.py

  scripts/
    download_models.py
    verify_model_checksums.py
    start_text_llamacpp.sh
    start_vision_service.sh
    start_asr_service.sh
    smoke_test_local.py

  tests/
```

---

## 17. Codex Usage Requirements

Codex shall be used as the engineering agent to build the repository incrementally.

Codex shall **not** be treated as the model trainer. It will generate and maintain:

```text
- repo skeleton
- Gradio UI
- router
- schemas
- dataset registry
- synthetic data generator
- training scripts
- eval harness
- llama.cpp export scripts
- tests
- README
- demo assets
```

Each Codex task must follow this workflow:

```text
1. Inspect existing files.
2. Propose a concise implementation plan.
3. Implement one module only.
4. Add tests.
5. Run tests.
6. Summarize changed files and remaining gaps.
```

---

## 18. Codex Build Sequence

Use Codex in this order:

```text
1. Create AGENTS.md
2. Create repository skeleton
3. Build Gradio app shell
4. Build task router
5. Build document/output schemas
6. Build dataset registry
7. Build synthetic document generator
8. Build local tools: PDF/image/Excel/TXT/PDF export
9. Build text model client
10. Build vision model client
11. Build ASR client
12. Build eval harness
13. Build Nemotron training loop
14. Build MiniCPM-V training loop
15. Build Parakeet eval loop
16. Build llama.cpp export/startup scripts
17. Build final Hugging Face Space packaging
18. Build demo script and sample files
```

---

## 19. Key Codex Prompts

### 19.1 AGENTS.md Prompt

```text
Create AGENTS.md for local-lm.

Project:
local-lm is a Gradio-based local-first small-model assistant for elderly and accessibility-constrained users, built for the Hugging Face Build Small Hackathon under the Backyard AI track.

Constraints:
- App must be a Gradio app hosted as a Hugging Face Space.
- Total deployed model parameters must be <=32B.
- Models must come only from OpenBMB, NVIDIA, Cohere, or BFL.
- v1 models:
  - nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 for training
  - nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF for deployment
  - openbmb/MiniCPM-V-4.6 for vision/OCR/image tasks
  - nvidia/parakeet-tdt-0.6b-v3 for ASR
- Use llama.cpp/GGUF wherever possible.
- No cloud inference.
- No cloud OCR.
- No remote telemetry.
- privacy_mode=strict.
- allow_web=false by default.
- User files must not be uploaded to third-party APIs.
- Unknown-license datasets must be rejected.
- Real financial, medical, legal, or identity documents are blocked unless synthetic, redacted, or explicit opt-in.

Engineering standards:
- Python 3.11+
- Gradio UI
- Pydantic schemas
- pytest
- ruff
- openpyxl
- SQLite FTS5
- local-only model clients

Before coding:
- inspect files
- propose concise plan
- implement one module
- add tests
- run tests
- summarize changed files and gaps.
```

### 19.2 Gradio App Prompt

```text
Build the Gradio app shell for local-lm.

Requirements:
- Accessible high-contrast UI
- Large buttons
- Task tabs:
  1. Read / Summarize
  2. Convert Document
  3. Describe Image
  4. Translate Image Text
  5. Speech to Text
  6. Privacy / Settings
- Each tab should have sample inputs and clear outputs.
- Do not call models yet; use placeholder functions.
- Add custom CSS.
- Add tests for app import and route wiring.
```

### 19.3 Router Prompt

```text
Implement deterministic task routing.

Inputs:
- user text
- uploaded files
- selected UI mode

Routes:
- summarize_text
- summarize_url
- document_to_excel
- describe_image
- translate_image_text
- speech_to_text
- general_assistant

Requirements:
- No model call required for routing.
- Manual UI mode overrides automatic routing.
- Return route, reason, required model, required tools, and safety flags.
- Add pytest coverage.
```

### 19.4 Dataset Registry Prompt

```text
Implement dataset registry and acceptance gate.

Sources:
- Awesome Public Datasets
- AWS Registry of Open Data
- Azure Open Datasets
- UCI ML Repository
- Google Dataset Search
- Google Cloud Marketplace free datasets
- Wikimedia
- Hugging Face datasets
- synthetic
- manual manifests

Rules:
- Unknown license means reject.
- Non-commercial license means eval-only unless explicitly configured.
- High-PII data blocked unless redacted or opt-in.
- Every dataset must have region, country, language, task, modality, license, and PII metadata.

Outputs:
- data/registry/dataset_candidates.jsonl
- data/registry/approved_datasets.jsonl
- data/registry/rejected_datasets.jsonl
- reports/dataset_registry_audit.md
```

### 19.5 Synthetic Documents Prompt

```text
Implement synthetic regional document generation for invoices, receipts, bills, bank statements, and handwritten notes.

Regions:
- India
- Southeast Asia
- North America
- Europe

Generate:
- PNG
- optional PDF
- ground-truth JSON
- expected Excel rows
- metadata JSONL

Add noise:
- skew
- blur
- low contrast
- shadows
- mobile camera perspective
- partial crop

All PII must be synthetic.
Add tests for numeric reconciliation.
```

### 19.6 Training Loop Prompt

```text
Build LoRA/QLoRA training loops.

Text:
- nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- train for routing, summarization, tool-call JSON, JSON repair, elderly-safe explanations

Vision:
- openbmb/MiniCPM-V-4.6
- train for OCR, document extraction, bank statements, handwritten notes, image description, image translation

ASR:
- nvidia/parakeet-tdt-0.6b-v3
- eval-first; Indian/SEA ASR experimental

Requirements:
- dry-run mode
- local logs only
- no wandb by default
- eval reports in JSON and Markdown
```

### 19.7 Eval Harness Prompt

```text
Build local eval harness.

Evaluate:
- base model
- LoRA adapter
- merged model
- quantized GGUF model
- local runtime endpoint

Tasks:
- route_task
- summarize_text
- invoice_to_json
- invoice_to_excel
- bank_statement_to_transactions
- handwritten_note_to_text
- describe_image
- translate_image_text
- speech_to_text

Critical failures:
- invalid JSON
- hallucinated financial totals
- missing human-review flag
- unsafe medical/legal/financial certainty
- identity guessing from image
- unsupported-language hallucination
- cloud call attempted
```

---

## 20. Milestones

### Milestone 1 — Hackathon Skeleton

```text
- Gradio Space boots
- task UI works
- router works
- sample outputs mocked
- README explains concept
```

### Milestone 2 — Local Tools

```text
- image preprocessing
- PDF extraction
- Excel export
- JSON/TXT/PDF export
- safety warnings
```

### Milestone 3 — Model Integration

```text
- Nemotron GGUF text inference
- MiniCPM-V image inference
- Parakeet ASR inference
- model manifest
- parameter count display
```

### Milestone 4 — Dataset and Training Assets

```text
- dataset registry
- synthetic document generation
- LoRA training scripts
- eval harness
- sample fine-tuned adapter if feasible
```

### Milestone 5 — Final Demo

```text
- Space hosted
- sample files included
- demo video recorded
- social post prepared
- field notes/report published
```

---

## 21. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| HF Space resource limits | App may be slow | Use quantized models, smaller defaults, sample-mode demo |
| MiniCPM-V GGUF deployment friction | Vision path may fail in Space | Keep Transformers fallback |
| `llama.cpp` setup in Space | Build complexity | Use prebuilt wheel/binary if possible; document fallback |
| ASR language gap | India/SEA voice support weak | Mark experimental, support English first |
| OCR hallucination | Unsafe document exports | Schema validation, confidence flags, human review |
| Bank-statement errors | Financial harm risk | Always require review |
| Dataset licensing | Disqualification/legal risk | Dataset gate and dataset cards |
| Too many features | Incomplete hackathon demo | Prioritize 3 demo flows: summarize, invoice-to-Excel, describe image |

---

## 22. MVP Recommendation

For hackathon submission, prioritize **three polished flows**:

```text
1. Describe Image
   - immediate accessibility value
   - strong Backyard AI fit

2. Convert Bill/Invoice to Excel
   - clear practical utility
   - impressive demo

3. Summarize Article/Wikipedia
   - text model utility
   - simple to demonstrate
```

Speech-to-text should be included as a basic feature, but not treated as the central demo unless it is stable.

---

## 23. Definition of Done

The project is done for hackathon submission when:

```text
- Hugging Face Space launches successfully
- App is built with Gradio
- Model budget is documented and under 32B
- No external inference API is required
- At least three core flows work end-to-end
- Sample files are provided
- Export to JSON/TXT/XLSX works
- Safety warnings appear for financial documents
- README explains:
  - architecture
  - models
  - parameter count
  - local-first behavior
  - limitations
  - dataset/training approach
- Demo video is recorded
- Social post is prepared
```

---

## 24. Final Product Statement

```text
local-lm is a Gradio-based, local-first, small-model accessibility assistant for elderly and low-vision users. It uses a routed bundle of small models under 32B total parameters to summarize information, read documents, convert bills to structured files, describe images, translate visible text, and transcribe speech without relying on cloud inference or cloud OCR.
```

The project’s strongest hackathon angle is:

```text
A real-world Backyard AI assistant for someone’s parent or grandparent:
simple, private, local, accessible, and small enough to fit the hackathon constraint.
```
