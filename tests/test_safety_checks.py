from __future__ import annotations

import pytest

from services.tools.safety_checks import (
    assert_no_cloud_runtime,
    sensitive_categories_for_text,
    warnings_for_output,
)


@pytest.mark.parametrize("category", ["financial", "medical", "legal"])
def test_sensitive_outputs_include_uncertainty_and_human_review_warning(category: str) -> None:
    warnings = warnings_for_output(category)

    assert len(warnings) == 1
    assert warnings[0].requires_human_review is True
    assert "uncertain" in warnings[0].message
    assert "qualified human" in warnings[0].message


def test_sensitive_category_detection_for_user_text() -> None:
    categories = sensitive_categories_for_text(
        "Can you explain this medical bill and legal notice?"
    )

    assert categories == ["financial", "legal", "medical"]


def test_no_cloud_runtime_settings_pass_for_strict_local_defaults() -> None:
    assert_no_cloud_runtime(
        {
            "local_only": True,
            "privacy_mode": "strict",
            "allow_web": False,
            "allow_remote_inference": False,
            "allow_remote_file_uploads": False,
            "allow_external_apis": False,
            "telemetry_enabled": False,
        }
    )


def test_no_cloud_runtime_rejects_web_or_telemetry() -> None:
    with pytest.raises(ValueError, match="allow_web"):
        assert_no_cloud_runtime({"privacy_mode": "strict", "allow_web": True})

    with pytest.raises(ValueError, match="telemetry_enabled"):
        assert_no_cloud_runtime({"privacy_mode": "strict", "telemetry_enabled": True})
