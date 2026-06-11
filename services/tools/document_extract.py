from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from services.gateway.schemas import (
    BankStatementExtractionOutput,
    BankTransaction,
    DocumentExtractionOutput,
    InvoiceExtractionOutput,
)
from services.tools.pdf_extract import extract_pdf_placeholder

MONEY_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_local_document(
    path: Path,
) -> InvoiceExtractionOutput | BankStatementExtractionOutput | DocumentExtractionOutput:
    suffix = path.suffix.casefold()
    if suffix != ".txt":
        return extract_pdf_placeholder("local_document")
    text = path.read_text(encoding="utf-8")
    normalized = text.casefold()
    if "bank statement" in normalized or "transactions:" in normalized:
        return extract_bank_statement_text(text)
    if "invoice" in normalized or "receipt" in normalized or "subtotal:" in normalized:
        return extract_invoice_text(text)
    return DocumentExtractionOutput(
        document_type="text_document",
        fields={"text_length": len(text)},
        raw_text=text,
        confidence=0.4,
        warnings=["Generic text document extraction only; review before using."],
        human_review_required=True,
    )


def extract_vision_document_json(
    model_text: str,
) -> InvoiceExtractionOutput | BankStatementExtractionOutput | DocumentExtractionOutput:
    payload = _extract_json_object(model_text)
    document_type = str(payload.get("document_type", "")).strip().casefold()
    if not document_type:
        raise ValueError("vision document output missing document_type")

    if document_type in {"invoice", "bill", "receipt"}:
        return InvoiceExtractionOutput.model_validate(_financial_payload_defaults(payload))
    if document_type == "bank_statement":
        return BankStatementExtractionOutput.model_validate(_financial_payload_defaults(payload))
    return DocumentExtractionOutput.model_validate(_generic_payload_defaults(payload))


def extraction_rows(
    extraction: InvoiceExtractionOutput | BankStatementExtractionOutput | DocumentExtractionOutput,
) -> list[dict[str, Any]]:
    if isinstance(extraction, BankStatementExtractionOutput):
        return [
            {
                "date": transaction.date,
                "description": transaction.description,
                "debit": transaction.debit,
                "credit": transaction.credit,
                "balance": transaction.balance,
                "confidence": transaction.confidence,
                "needs_review": transaction.needs_review,
            }
            for transaction in extraction.transactions
        ]
    if isinstance(extraction, InvoiceExtractionOutput):
        rows = [
            {"field": "document_type", "value": extraction.document_type},
            {"field": "currency", "value": extraction.currency},
            {"field": "subtotal", "value": extraction.subtotal},
            {"field": "tax_amount", "value": extraction.tax_amount},
            {"field": "total", "value": extraction.total},
            {"field": "confidence", "value": extraction.confidence},
            {"field": "human_review_required", "value": extraction.human_review_required},
        ]
        rows.extend(
            {"field": key, "value": value} for key, value in sorted(extraction.fields.items())
        )
        for index, item in enumerate(extraction.line_items, start=1):
            rows.append({"field": f"line_item_{index}", "value": item})
        return rows
    return [{"field": key, "value": value} for key, value in extraction.fields.items()]


def _extract_json_object(model_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(model_text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(model_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("vision document output did not contain a JSON object")


def _financial_payload_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("fields", {})
    normalized.setdefault("raw_ocr_text", "")
    normalized.setdefault("confidence", 0.6)
    normalized["human_review_required"] = True
    warnings = normalized.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    warnings.append("Local vision model extraction; verify financial fields before use.")
    normalized["warnings"] = warnings
    return normalized


def _generic_payload_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("fields", {})
    normalized.setdefault("raw_text", normalized.get("raw_ocr_text", ""))
    normalized.setdefault("confidence", 0.4)
    warnings = normalized.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    warnings.append("Local vision model extraction; review before use.")
    normalized["warnings"] = warnings
    if normalized["confidence"] < 0.7:
        normalized["human_review_required"] = True
    return normalized


def extract_invoice_text(text: str) -> InvoiceExtractionOutput:
    fields: dict[str, Any] = {}
    for label, field_name in {
        "Vendor": "vendor",
        "Region": "region",
        "Invoice number": "invoice_number",
        "GSTIN": "gstin",
        "UPI reference": "upi_reference",
    }.items():
        value = _line_value(text, label)
        if value:
            fields[field_name] = value

    line_items = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        amount = _money_value(stripped)
        quantity_match = re.search(r"quantity\s+(\d+)", stripped, flags=re.IGNORECASE)
        hsn_match = re.search(r"HSN/SAC\s+([A-Za-z0-9-]+)", stripped, flags=re.IGNORECASE)
        description = stripped[2:].split(",")[0].strip()
        line_items.append(
            {
                "description": description,
                "hsn_sac": hsn_match.group(1) if hsn_match else "",
                "quantity": int(quantity_match.group(1)) if quantity_match else 1,
                "amount": amount,
            }
        )

    subtotal = _money_value(_line_value(text, "Subtotal"))
    total = _money_value(_line_value(text, "Total"))
    tax_amount = _sum_tax_lines(text)
    currency = _currency_from_text(text)
    warnings = ["Synthetic/local text extraction; verify before use."]
    return InvoiceExtractionOutput(
        document_type="invoice",
        fields=fields,
        line_items=line_items,
        currency=currency,
        subtotal=subtotal,
        tax_amount=tax_amount,
        total=total,
        raw_ocr_text=text,
        confidence=0.85,
        warnings=warnings,
        human_review_required=True,
    )


def extract_bank_statement_text(text: str) -> BankStatementExtractionOutput:
    transactions: list[BankTransaction] = []
    warnings = ["Bank-statement extraction always requires human review."]
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5 or not re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
            continue
        credit = _labeled_money(parts[2], "credit")
        debit = _labeled_money(parts[3], "debit")
        balance = _money_value(parts[4])
        transactions.append(
            BankTransaction(
                date=parts[0],
                description=parts[1],
                credit=credit,
                debit=debit,
                balance=balance,
                confidence=0.85,
                needs_review=True,
            )
        )

    reconciliation_warning = _bank_reconciliation_warning(transactions)
    if reconciliation_warning:
        warnings.append(reconciliation_warning)

    return BankStatementExtractionOutput(
        transactions=transactions,
        currency=_currency_from_text(text),
        raw_ocr_text=text,
        confidence=0.82 if transactions else 0.2,
        warnings=warnings,
        human_review_required=True,
    )


def _line_value(text: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", flags=re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _money_value(text: str) -> float | None:
    matches = MONEY_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return round(float(matches[-1]), 2)


def _labeled_money(text: str, label: str) -> float:
    if label.casefold() not in text.casefold():
        return 0.0
    value = _money_value(text)
    return value if value is not None else 0.0


def _sum_tax_lines(text: str) -> float | None:
    total = 0.0
    found = False
    for line in text.splitlines():
        lower = line.casefold()
        is_tax_line = any(
            marker in lower for marker in ("tax", "vat", "cgst", "sgst", "igst", "sales tax")
        )
        if is_tax_line and ":" in line and "gstin" not in lower:
            if lower.startswith("total"):
                continue
            value = _money_value(line)
            if value is not None:
                total += value
                found = True
    return round(total, 2) if found else None


def _currency_from_text(text: str) -> str:
    for currency in (
        "INR",
        "USD",
        "CAD",
        "EUR",
        "GBP",
        "CHF",
        "SGD",
        "MYR",
        "IDR",
        "THB",
        "PHP",
        "VND",
    ):
        if currency in text:
            return currency
    return ""


def _bank_reconciliation_warning(transactions: list[BankTransaction]) -> str:
    if len(transactions) < 2:
        return ""
    previous = transactions[0].balance
    if previous is None:
        return ""
    for transaction in transactions[1:]:
        if transaction.balance is None:
            continue
        expected = round(previous + transaction.credit - transaction.debit, 2)
        actual = round(transaction.balance, 2)
        if expected != actual:
            return "Running balance did not reconcile; review transaction rows."
        previous = transaction.balance
    return ""
