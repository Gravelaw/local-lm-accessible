from __future__ import annotations

from data.schemas.task_mapping import map_candidate_to_tasks
from scripts.registry_common import seed_records


def _seed(dataset_id: str):
    for record in seed_records():
        if record.dataset_id == dataset_id:
            return record
    raise AssertionError(f"missing seed record: {dataset_id}")


def test_sroie_maps_to_receipt_extraction_and_document_ocr() -> None:
    mapping = map_candidate_to_tasks(_seed("huggingface:ryanznie/SROIE_2019_with_labels"))

    assert "receipt_extraction" in mapping.mapped_tasks
    assert "document_ocr" in mapping.mapped_tasks


def test_cord_maps_to_receipt_extraction_document_ocr_and_southeast_asia() -> None:
    candidate = _seed("manual:CORD")
    mapping = map_candidate_to_tasks(candidate)

    assert "Southeast Asia" in candidate.regions
    assert "receipt_extraction" in mapping.mapped_tasks
    assert "document_ocr" in mapping.mapped_tasks


def test_vizwiz_maps_to_accessibility_description() -> None:
    mapping = map_candidate_to_tasks(_seed("manual:VizWiz-VQA"))

    assert "image_accessibility_description" in mapping.mapped_tasks


def test_textvqa_maps_to_image_translation_and_vqa() -> None:
    mapping = map_candidate_to_tasks(_seed("huggingface:facebook/textvqa"))

    assert "image_text_translation" in mapping.mapped_tasks
    assert "visual_question_answering" in mapping.mapped_tasks


def test_indicvoices_maps_to_speech_to_text_eval_or_experimental() -> None:
    mapping = map_candidate_to_tasks(_seed("huggingface:ai4bharat/IndicVoices"), "eval_only")

    assert "speech_to_text" in mapping.mapped_tasks
    assert mapping.eval_only is True


def test_uci_online_retail_maps_to_tabular_reasoning_eval_only() -> None:
    mapping = map_candidate_to_tasks(_seed("uci:online-retail"))

    assert "tabular_reasoning" in mapping.mapped_tasks
    assert "eval_only" in mapping.mapped_tasks
    assert mapping.eval_only is True


def test_epo_maps_to_technical_summarization_eval_only() -> None:
    mapping = map_candidate_to_tasks(_seed("epo:ops-patent-data"))

    assert "technical_summarization" in mapping.mapped_tasks
    assert "eval_only" in mapping.mapped_tasks
    assert mapping.eval_only is True
