from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

from babel.dates import format_date
from babel.numbers import format_currency
from faker import Faker
from jinja2 import Template
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.tools.image_preprocess import AugmentationConfig, apply_augmentations  # noqa: E402

DocumentKind = Literal["invoice", "receipt", "bank_statement", "handwritten_note"]

SUPPORTED_REGIONS = ("India", "Southeast Asia", "North America", "Europe")
TWO_PLACES = Decimal("0.01")
SYNTHETIC_DATASET_REGISTRY_NAME = "synthetic_dataset_candidates.jsonl"


@dataclass(frozen=True)
class RegionProfile:
    region: str
    country: str
    locale: str
    currency: str
    tax_label: str
    tax_rate: Decimal
    languages: tuple[str, ...]
    labels: dict[str, str]
    secondary_tax_label: str | None = None
    secondary_tax_rate: Decimal = Decimal("0.00")
    decimal_comma: bool = False


REGION_PROFILES: dict[str, tuple[RegionProfile, ...]] = {
    "India": (
        RegionProfile(
            region="India",
            country="India",
            locale="en_IN",
            currency="INR",
            tax_label="CGST",
            tax_rate=Decimal("0.09"),
            secondary_tax_label="SGST",
            secondary_tax_rate=Decimal("0.09"),
            languages=("English", "Hindi", "Tamil", "Bengali", "Marathi", "Telugu"),
            labels={
                "invoice": "Invoice / चालान / விலைப்பட்டியல் / চালান / बीजक / ఇన్వాయిస్",
                "receipt": "Receipt / रसीद / ரசீது / রসিদ / पावती / రసీదు",
                "tax_id": "GSTIN",
                "tax_code": "HSN/SAC",
                "payment_ref": "UPI reference",
            },
        ),
        RegionProfile(
            region="India",
            country="India",
            locale="hi_IN",
            currency="INR",
            tax_label="IGST",
            tax_rate=Decimal("0.18"),
            languages=("English", "Hindi", "Tamil", "Bengali", "Marathi", "Telugu"),
            labels={
                "invoice": "Service Invoice / सेवा चालान",
                "receipt": "Merchant Receipt / व्यापारी रसीद",
                "tax_id": "GSTIN",
                "tax_code": "SAC",
                "payment_ref": "UPI reference",
            },
        ),
    ),
    "Southeast Asia": (
        RegionProfile(
            region="Southeast Asia",
            country="Singapore",
            locale="en_SG",
            currency="SGD",
            tax_label="GST",
            tax_rate=Decimal("0.09"),
            languages=("English", "Chinese"),
            labels={
                "invoice": "Tax Invoice / 发票",
                "receipt": "Receipt / 收据",
                "tax_id": "Singapore GST Reg No",
                "tax_code": "Item code",
                "payment_ref": "PayNow reference",
            },
        ),
        RegionProfile(
            region="Southeast Asia",
            country="Malaysia",
            locale="ms_MY",
            currency="MYR",
            tax_label="SST",
            tax_rate=Decimal("0.08"),
            languages=("English", "Malay"),
            labels={
                "invoice": "Invoice / Invois",
                "receipt": "Receipt / Resit",
                "tax_id": "SST No",
                "tax_code": "Service code",
                "payment_ref": "DuitNow reference",
            },
        ),
        RegionProfile(
            region="Southeast Asia",
            country="Indonesia",
            locale="id_ID",
            currency="IDR",
            tax_label="PPN/VAT",
            tax_rate=Decimal("0.11"),
            languages=("Indonesian", "English"),
            labels={
                "invoice": "Faktur / Invoice",
                "receipt": "Kwitansi / Receipt",
                "tax_id": "NPWP-like synthetic ID",
                "tax_code": "Kode barang",
                "payment_ref": "QRIS reference",
            },
        ),
        RegionProfile(
            region="Southeast Asia",
            country="Thailand",
            locale="th_TH",
            currency="THB",
            tax_label="VAT",
            tax_rate=Decimal("0.07"),
            languages=("Thai", "English"),
            labels={
                "invoice": "Tax Invoice / ใบกำกับภาษี",
                "receipt": "Receipt / ใบเสร็จ",
                "tax_id": "Thai Tax ID",
                "tax_code": "Item code",
                "payment_ref": "PromptPay reference",
            },
        ),
        RegionProfile(
            region="Southeast Asia",
            country="Philippines",
            locale="en_PH",
            currency="PHP",
            tax_label="VAT",
            tax_rate=Decimal("0.12"),
            languages=("English", "Filipino"),
            labels={
                "invoice": "Invoice / Resibo",
                "receipt": "Receipt / Resibo",
                "tax_id": "TIN-like synthetic ID",
                "tax_code": "Item code",
                "payment_ref": "GCash reference",
            },
        ),
        RegionProfile(
            region="Southeast Asia",
            country="Vietnam",
            locale="vi_VN",
            currency="VND",
            tax_label="VAT",
            tax_rate=Decimal("0.10"),
            languages=("Vietnamese", "English"),
            labels={
                "invoice": "Hóa đơn / Invoice",
                "receipt": "Biên lai / Receipt",
                "tax_id": "MST-like synthetic ID",
                "tax_code": "Mã hàng",
                "payment_ref": "Bank transfer reference",
            },
        ),
    ),
    "North America": (
        RegionProfile(
            region="North America",
            country="United States",
            locale="en_US",
            currency="USD",
            tax_label="Sales tax",
            tax_rate=Decimal("0.0825"),
            languages=("English",),
            labels={
                "invoice": "Service Invoice",
                "receipt": "Merchant Receipt",
                "tax_id": "Synthetic EIN",
                "tax_code": "Service code",
                "payment_ref": "Masked card",
            },
        ),
        RegionProfile(
            region="North America",
            country="Canada",
            locale="en_CA",
            currency="CAD",
            tax_label="GST/HST",
            tax_rate=Decimal("0.13"),
            secondary_tax_label="PST",
            secondary_tax_rate=Decimal("0.00"),
            languages=("English", "French"),
            labels={
                "invoice": "Invoice / Facture",
                "receipt": "Receipt / Reçu",
                "tax_id": "Business number",
                "tax_code": "Service code",
                "payment_ref": "Masked card",
            },
        ),
    ),
    "Europe": (
        RegionProfile(
            region="Europe",
            country="Germany",
            locale="de_DE",
            currency="EUR",
            tax_label="VAT / MwSt",
            tax_rate=Decimal("0.19"),
            languages=("English", "German"),
            labels={
                "invoice": "Invoice / Rechnung",
                "receipt": "Receipt / Quittung",
                "tax_id": "VAT ID",
                "tax_code": "Item code",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
        RegionProfile(
            region="Europe",
            country="France",
            locale="fr_FR",
            currency="EUR",
            tax_label="TVA",
            tax_rate=Decimal("0.20"),
            languages=("English", "French"),
            labels={
                "invoice": "Invoice / Facture",
                "receipt": "Receipt / Reçu",
                "tax_id": "VAT ID",
                "tax_code": "Code article",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
        RegionProfile(
            region="Europe",
            country="United Kingdom",
            locale="en_GB",
            currency="GBP",
            tax_label="VAT",
            tax_rate=Decimal("0.20"),
            languages=("English",),
            labels={
                "invoice": "VAT Invoice",
                "receipt": "Receipt",
                "tax_id": "VAT ID",
                "tax_code": "Item code",
                "payment_ref": "IBAN",
            },
        ),
        RegionProfile(
            region="Europe",
            country="Switzerland",
            locale="de_CH",
            currency="CHF",
            tax_label="VAT / MWST",
            tax_rate=Decimal("0.081"),
            languages=("German", "French", "Italian"),
            labels={
                "invoice": "Invoice / Rechnung / Facture / Fattura",
                "receipt": "Receipt / Quittung",
                "tax_id": "VAT ID",
                "tax_code": "Item code",
                "payment_ref": "IBAN",
            },
        ),
        RegionProfile(
            region="Europe",
            country="Spain",
            locale="es_ES",
            currency="EUR",
            tax_label="IVA",
            tax_rate=Decimal("0.21"),
            languages=("Spanish", "English"),
            labels={
                "invoice": "Invoice / Factura",
                "receipt": "Receipt / Recibo",
                "tax_id": "VAT ID",
                "tax_code": "Código",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
        RegionProfile(
            region="Europe",
            country="Italy",
            locale="it_IT",
            currency="EUR",
            tax_label="IVA",
            tax_rate=Decimal("0.22"),
            languages=("Italian", "English"),
            labels={
                "invoice": "Invoice / Fattura",
                "receipt": "Receipt / Ricevuta",
                "tax_id": "VAT ID",
                "tax_code": "Codice",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
        RegionProfile(
            region="Europe",
            country="Netherlands",
            locale="nl_NL",
            currency="EUR",
            tax_label="BTW",
            tax_rate=Decimal("0.21"),
            languages=("Dutch", "English"),
            labels={
                "invoice": "Invoice / Factuur",
                "receipt": "Receipt / Bon",
                "tax_id": "VAT ID",
                "tax_code": "Artikelcode",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
        RegionProfile(
            region="Europe",
            country="Portugal",
            locale="pt_PT",
            currency="EUR",
            tax_label="IVA",
            tax_rate=Decimal("0.23"),
            languages=("Portuguese", "English"),
            labels={
                "invoice": "Invoice / Fatura",
                "receipt": "Receipt / Recibo",
                "tax_id": "VAT ID",
                "tax_code": "Código",
                "payment_ref": "IBAN",
            },
            decimal_comma=True,
        ),
    ),
}


HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{{ title }}</title></head>
<body>
<h1>{{ title }}</h1>
<p>{{ merchant_name }}<br>{{ merchant_address }}<br>{{ contact }}</p>
<p>{{ tax_id_label }}: {{ tax_id }}<br>{{ payment_label }}: {{ payment_reference }}</p>
<table>
{% for row in rows -%}
  <tr><td>{{ row.label }}</td><td>{{ row.value }}</td></tr>
{% endfor -%}
</table>
</body>
</html>
"""
)


def money(value: Decimal) -> Decimal:
    return value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def synthetic_digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(count))


def synthetic_tax_id(profile: RegionProfile, rng: random.Random) -> str:
    if profile.region == "India":
        return f"SYN{synthetic_digits(rng, 2)}ABCDE{synthetic_digits(rng, 4)}F1Z{rng.randint(1, 9)}"
    if profile.region == "Europe":
        return f"SYN-{profile.country[:2].upper()}-VAT-{synthetic_digits(rng, 9)}"
    if profile.country == "United States":
        return f"SYN-EIN-{synthetic_digits(rng, 2)}-{synthetic_digits(rng, 7)}"
    return f"SYN-{profile.country[:3].upper()}-TAX-{synthetic_digits(rng, 8)}"


def synthetic_payment_reference(profile: RegionProfile, rng: random.Random) -> str:
    if profile.region == "India":
        return f"UPI-SYN-{synthetic_digits(rng, 12)}"
    if profile.region == "Europe":
        return f"SYNIBAN{synthetic_digits(rng, 18)}"
    if profile.region == "North America":
        return f"**** **** **** {synthetic_digits(rng, 4)}"
    return f"SYN-PAY-{synthetic_digits(rng, 12)}"


def synthetic_contact(rng: random.Random) -> str:
    phone = f"+99-000-{synthetic_digits(rng, 4)}-{synthetic_digits(rng, 4)}"
    email = f"hello{rng.randint(100, 999)}@synthetic.invalid"
    return f"{phone} | {email}"


def choose_profile(region: str, rng: random.Random) -> RegionProfile:
    if region not in REGION_PROFILES:
        raise ValueError(f"unsupported region: {region}")
    return rng.choice(REGION_PROFILES[region])


def format_money(profile: RegionProfile, amount: Decimal) -> str:
    return format_currency(amount, profile.currency, locale=profile.locale)


def format_doc_date(profile: RegionProfile, value: date) -> str:
    return format_date(value, format="medium", locale=profile.locale)


def build_line_items(
    profile: RegionProfile,
    rng: random.Random,
    count: int,
) -> list[dict[str, Any]]:
    item_names = [
        "large print support session",
        "home delivery assistance",
        "document reading service",
        "assistive setup visit",
        "accessible account summary",
        "community transport booking",
    ]
    rows = []
    for index in range(count):
        quantity = Decimal(rng.randint(1, 3))
        unit_price = money(Decimal(rng.randint(500, 9500)) / Decimal("100"))
        total = money(quantity * unit_price)
        rows.append(
            {
                "description": item_names[(index + rng.randint(0, 3)) % len(item_names)],
                "quantity": int(quantity),
                "unit_price": str(unit_price),
                "line_total": str(total),
                "tax_code": tax_code(profile, rng),
            }
        )
    return rows


def tax_code(profile: RegionProfile, rng: random.Random) -> str:
    if profile.region == "India":
        return str(rng.choice(["9983", "8517", "4901", "SAC-00440410"]))
    return f"SYN-{rng.randint(1000, 9999)}"


def calculate_invoice_totals(profile: RegionProfile, items: list[dict[str, Any]]) -> dict[str, str]:
    subtotal = money(sum(Decimal(item["line_total"]) for item in items))
    primary_tax = money(subtotal * profile.tax_rate)
    secondary_tax = money(subtotal * profile.secondary_tax_rate)
    total = money(subtotal + primary_tax + secondary_tax)
    return {
        "subtotal": str(subtotal),
        "tax_rate": str(profile.tax_rate),
        "tax_label": profile.tax_label,
        "tax_amount": str(primary_tax),
        "secondary_tax_label": profile.secondary_tax_label or "",
        "secondary_tax_rate": str(profile.secondary_tax_rate),
        "secondary_tax_amount": str(secondary_tax),
        "total": str(total),
    }


def build_invoice_record(
    document_kind: DocumentKind,
    profile: RegionProfile,
    rng: random.Random,
    index: int,
) -> dict[str, Any]:
    fake = Faker("en_US")
    fake.seed_instance(rng.randint(1, 2_000_000))
    items = build_line_items(profile, rng, rng.randint(2, 5))
    totals = calculate_invoice_totals(profile, items)
    issued_on = date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))
    merchant_name = f"{fake.company()} Synthetic Services"
    customer_name = f"Synthetic Customer {index:04d}"
    document_id = f"{document_kind}-{profile.region.lower().replace(' ', '-')}-{index:04d}"
    return {
        "document_id": document_id,
        "document_type": document_kind,
        "region": profile.region,
        "country": profile.country,
        "locale": profile.locale,
        "currency": profile.currency,
        "languages": list(profile.languages),
        "title": profile.labels["receipt" if document_kind == "receipt" else "invoice"],
        "merchant_name": merchant_name,
        "merchant_address": f"Synthetic Block {rng.randint(10, 999)}, {profile.country}",
        "customer_name": customer_name,
        "contact": synthetic_contact(rng),
        "tax_id_label": profile.labels["tax_id"],
        "tax_id": synthetic_tax_id(profile, rng),
        "tax_code_label": profile.labels["tax_code"],
        "payment_label": profile.labels["payment_ref"],
        "payment_reference": synthetic_payment_reference(profile, rng),
        "issued_on": issued_on.isoformat(),
        "issued_on_display": format_doc_date(profile, issued_on),
        "items": items,
        "totals": totals,
        "formatted_total": format_money(profile, Decimal(totals["total"])),
        "synthetic": True,
        "pii": "synthetic",
    }


def build_bank_statement_record(
    profile: RegionProfile,
    rng: random.Random,
    index: int,
) -> dict[str, Any]:
    fake = Faker("en_US")
    fake.seed_instance(rng.randint(1, 2_000_000))
    document_id = f"bank-statement-{profile.region.lower().replace(' ', '-')}-{index:04d}"
    start = money(Decimal(rng.randint(50_000, 300_000)) / Decimal("100"))
    balance = start
    transactions = []
    current_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 30))
    for tx_index in range(rng.randint(6, 10)):
        is_credit = tx_index % 3 == 0
        amount = money(Decimal(rng.randint(500, 25_000)) / Decimal("100"))
        if is_credit:
            balance = money(balance + amount)
            debit = Decimal("0.00")
            credit = amount
        else:
            balance = money(balance - amount)
            debit = amount
            credit = Decimal("0.00")
        transactions.append(
            {
                "date": (current_date + timedelta(days=tx_index * 2)).isoformat(),
                "description": f"SYNTHETIC TX {tx_index + 1}",
                "debit": str(debit),
                "credit": str(credit),
                "balance": str(balance),
            }
        )
    return {
        "document_id": document_id,
        "document_type": "bank_statement",
        "region": profile.region,
        "country": profile.country,
        "locale": profile.locale,
        "currency": profile.currency,
        "languages": list(profile.languages),
        "title": "Synthetic Bank Statement",
        "account_holder": f"Synthetic Customer {index:04d}",
        "bank_name": f"{fake.company()} Synthetic Bank",
        "account_number": f"SYN-ACCT-****{synthetic_digits(rng, 4)}",
        "payment_reference": synthetic_payment_reference(profile, rng),
        "starting_balance": str(start),
        "ending_balance": str(balance),
        "transactions": transactions,
        "formatted_total": format_money(profile, balance),
        "synthetic": True,
        "pii": "synthetic",
    }


def build_note_record(profile: RegionProfile, rng: random.Random, index: int) -> dict[str, Any]:
    document_id = f"handwritten-note-{profile.region.lower().replace(' ', '-')}-{index:04d}"
    issued_on = date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))
    note_lines = [
        "Synthetic reminder: read the large-print invoice aloud.",
        f"Region: {profile.region}; Country: {profile.country}.",
        "No real names, accounts, phones, or addresses are present.",
        f"Reference: SYN-NOTE-{synthetic_digits(rng, 8)}.",
    ]
    return {
        "document_id": document_id,
        "document_type": "handwritten_note",
        "region": profile.region,
        "country": profile.country,
        "locale": profile.locale,
        "currency": profile.currency,
        "languages": list(profile.languages),
        "title": "Synthetic Handwritten Note",
        "issued_on": issued_on.isoformat(),
        "issued_on_display": format_doc_date(profile, issued_on),
        "note_lines": note_lines,
        "synthetic": True,
        "pii": "synthetic",
    }


def render_rows(record: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {"label": "Document ID", "value": str(record["document_id"])},
        {"label": "Region", "value": str(record["region"])},
        {"label": "Country", "value": str(record["country"])},
    ]
    if record["document_type"] in {"invoice", "receipt"}:
        rows.extend(
            [
                {"label": "Date", "value": str(record["issued_on_display"])},
                {"label": "Merchant", "value": str(record["merchant_name"])},
                {"label": "Customer", "value": str(record["customer_name"])},
                {"label": str(record["tax_id_label"]), "value": str(record["tax_id"])},
                {"label": str(record["payment_label"]), "value": str(record["payment_reference"])},
                {"label": "Subtotal", "value": record["totals"]["subtotal"]},
                {
                    "label": str(record["totals"]["tax_label"]),
                    "value": record["totals"]["tax_amount"],
                },
                {"label": "Total", "value": record["totals"]["total"]},
            ]
        )
        if record["totals"]["secondary_tax_label"]:
            rows.insert(
                -1,
                {
                    "label": str(record["totals"]["secondary_tax_label"]),
                    "value": record["totals"]["secondary_tax_amount"],
                },
            )
    elif record["document_type"] == "bank_statement":
        rows.extend(
            [
                {"label": "Bank", "value": str(record["bank_name"])},
                {"label": "Account", "value": str(record["account_number"])},
                {"label": "Starting balance", "value": str(record["starting_balance"])},
                {"label": "Ending balance", "value": str(record["ending_balance"])},
            ]
        )
    else:
        rows.extend(
            {"label": f"Line {index + 1}", "value": line}
            for index, line in enumerate(record["note_lines"])
        )
    return rows


def raw_ocr_text(record: dict[str, Any]) -> str:
    lines = [str(record["title"]), f"Document ID: {record['document_id']}"]
    for row in render_rows(record):
        lines.append(f"{row['label']}: {row['value']}")
    if record["document_type"] in {"invoice", "receipt"}:
        lines.append("Items:")
        for item in record["items"]:
            lines.append(
                f"{item['description']} qty {item['quantity']} "
                f"code {item['tax_code']} total {item['line_total']}"
            )
    if record["document_type"] == "bank_statement":
        lines.append("Transactions:")
        for transaction in record["transactions"]:
            lines.append(
                " | ".join(
                    [
                        transaction["date"],
                        transaction["description"],
                        transaction["debit"],
                        transaction["credit"],
                        transaction["balance"],
                    ]
                )
            )
    return "\n".join(lines)


def make_html(record: dict[str, Any]) -> str:
    return HTML_TEMPLATE.render(
        title=record["title"],
        merchant_name=record.get("merchant_name", record.get("bank_name", "Synthetic")),
        merchant_address=record.get("merchant_address", f"Synthetic address, {record['country']}"),
        contact=record.get("contact", "synthetic contact only"),
        tax_id_label=record.get("tax_id_label", "Synthetic ID"),
        tax_id=record.get("tax_id", record.get("account_number", "SYN-ID")),
        payment_label=record.get("payment_label", "Reference"),
        payment_reference=record.get("payment_reference", "SYN-REF"),
        rows=render_rows(record),
    )


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans.ttf", size)


def render_png(record: dict[str, Any], path: Path, rng: random.Random, augment: bool) -> None:
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38)
    body_font = load_font(24)
    small_font = load_font(19)
    y = 48
    draw.text((56, y), str(record["title"]), fill=(25, 25, 25), font=title_font)
    y += 60
    for row in render_rows(record):
        draw.text((56, y), f"{row['label']}: {row['value']}", fill=(35, 35, 35), font=body_font)
        y += 38
    y += 16
    if record["document_type"] in {"invoice", "receipt"}:
        draw.text((56, y), "Items", fill=(20, 20, 20), font=body_font)
        y += 38
        for item in record["items"]:
            text = (
                f"{item['description']} | qty {item['quantity']} | "
                f"{record['tax_code_label']} {item['tax_code']} | {item['line_total']}"
            )
            draw.text((56, y), text, fill=(45, 45, 45), font=small_font)
            y += 32
    elif record["document_type"] == "bank_statement":
        draw.text((56, y), "Transactions", fill=(20, 20, 20), font=body_font)
        y += 38
        for transaction in record["transactions"]:
            text = (
                f"{transaction['date']} {transaction['description']} "
                f"D {transaction['debit']} C {transaction['credit']} B {transaction['balance']}"
            )
            draw.text((56, y), text, fill=(45, 45, 45), font=small_font)
            y += 32
    else:
        for line in record["note_lines"]:
            draw.text((70, y), line, fill=(30, 30, 90), font=body_font)
            y += 48
    draw.text((56, 1530), "SYNTHETIC DOCUMENT - NO REAL PII", fill=(110, 20, 20), font=small_font)
    if augment:
        image = apply_augmentations(
            image,
            AugmentationConfig(
                blur=rng.random() < 0.35,
                skew=rng.random() < 0.35,
                low_contrast=rng.random() < 0.35,
                shadows=rng.random() < 0.25,
                folds=rng.random() < 0.25,
                mobile_camera_angle=rng.random() < 0.25,
                partial_crop=rng.random() < 0.20,
                seed=rng.randint(1, 2_000_000),
            ),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_pdf(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    y = height - 48
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(48, y, str(record["title"])[:90])
    pdf.setFont("Helvetica", 10)
    y -= 28
    for line in raw_ocr_text(record).splitlines():
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.drawString(48, y, safe_line[:110])
        y -= 14
        if y < 48:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 48
    pdf.save()


def write_excel_rows(record: dict[str, Any], path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "expected_rows"
    sheet.append(["document_id", "label", "value"])
    for row in render_rows(record):
        sheet.append([record["document_id"], row["label"], row["value"]])
    if record["document_type"] in {"invoice", "receipt"}:
        for item in record["items"]:
            sheet.append([record["document_id"], item["description"], item["line_total"]])
    if record["document_type"] == "bank_statement":
        for transaction in record["transactions"]:
            sheet.append(
                [record["document_id"], transaction["description"], transaction["balance"]]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def write_document_outputs(
    record: dict[str, Any],
    output_dir: Path,
    rng: random.Random,
    augment: bool,
) -> None:
    document_dir = output_dir / record["document_type"] / record["document_id"]
    document_dir.mkdir(parents=True, exist_ok=True)
    render_png(record, document_dir / "rendered.png", rng, augment)
    write_pdf(record, document_dir / "document.pdf")
    (document_dir / "source.html").write_text(make_html(record), encoding="utf-8")
    (document_dir / "ground_truth.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (document_dir / "raw_ocr.txt").write_text(raw_ocr_text(record), encoding="utf-8")
    (document_dir / "expected_excel_rows.json").write_text(
        json.dumps(render_rows(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_excel_rows(record, document_dir / "expected_excel_rows.xlsx")


def metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    task = _task_for_document_type(str(record["document_type"]))
    return {
        "document_id": record["document_id"],
        "document_type": record["document_type"],
        "region": record["region"],
        "country": record["country"],
        "languages": record["languages"],
        "language": record["languages"][0] if record["languages"] else "unknown",
        "currency": record["currency"],
        "license": "CC0-1.0",
        "modality": "image",
        "task": task,
        "synthetic": True,
        "pii": "synthetic",
        "pii_status": "synthetic",
        "source_type": "synthetic",
        "outputs": {
            "png": f"{record['document_type']}/{record['document_id']}/rendered.png",
            "pdf": f"{record['document_type']}/{record['document_id']}/document.pdf",
            "html": f"{record['document_type']}/{record['document_id']}/source.html",
            "ground_truth": f"{record['document_type']}/{record['document_id']}/ground_truth.json",
            "raw_ocr": f"{record['document_type']}/{record['document_id']}/raw_ocr.txt",
            "excel_rows": (
                f"{record['document_type']}/{record['document_id']}/expected_excel_rows.xlsx"
            ),
        },
    }


def synthetic_dataset_candidate(record: dict[str, Any]) -> dict[str, Any]:
    metadata = metadata_record(record)
    task = str(metadata["task"])
    document_id = str(metadata["document_id"])
    return {
        "source_catalog": "local-lm synthetic regional documents",
        "dataset_name": f"synthetic {metadata['document_type']} {document_id}",
        "dataset_url": f"https://example.local/local-lm/synthetic/{document_id}",
        "modality": metadata["modality"],
        "candidate_tasks": [task],
        "regions": [metadata["region"]],
        "countries": [metadata["country"]],
        "languages": metadata["languages"],
        "license_name": metadata["license"],
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "intended_use": "training",
        "source_quality": "high",
        "notes": (
            "Synthetic local-lm generated document. "
            f"pii_status={metadata['pii_status']}; source_type={metadata['source_type']}; "
            f"task={task}; region={metadata['region']}; country={metadata['country']}; "
            f"language={metadata['language']}; license={metadata['license']}."
        ),
        "redacted": False,
        "explicit_user_opt_in": False,
        "metadata_only": True,
        "cloud_hosted": False,
    }


def append_metadata(output_dir: Path, records: list[dict[str, Any]]) -> None:
    metadata_path = output_dir / "metadata.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"metadata row must be an object at {metadata_path}:{line_number}"
                    )
                document_id = str(payload.get("document_id", "")).strip()
                if not document_id:
                    raise ValueError(
                        f"metadata row missing document_id at {metadata_path}:{line_number}"
                    )
                merged[document_id] = payload
    for record in records:
        metadata = metadata_record(record)
        merged[str(metadata["document_id"])] = metadata
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for document_id in sorted(merged):
            metadata_file.write(json.dumps(merged[document_id], sort_keys=True) + "\n")
    write_synthetic_dataset_registry(output_dir, records)


def write_synthetic_dataset_registry(output_dir: Path, records: list[dict[str, Any]]) -> None:
    registry_path = output_dir / SYNTHETIC_DATASET_REGISTRY_NAME
    merged: dict[str, dict[str, Any]] = {}
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as registry_file:
            for line_number, line in enumerate(registry_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"synthetic registry row must be an object at {registry_path}:{line_number}"
                    )
                dataset_name = str(payload.get("dataset_name", "")).strip()
                if not dataset_name:
                    raise ValueError(
                        "synthetic registry row missing dataset_name at "
                        f"{registry_path}:{line_number}"
                    )
                merged[dataset_name] = payload
    for record in records:
        candidate = synthetic_dataset_candidate(record)
        merged[str(candidate["dataset_name"])] = candidate
    with registry_path.open("w", encoding="utf-8") as registry_file:
        for dataset_name in sorted(merged):
            registry_file.write(json.dumps(merged[dataset_name], sort_keys=True) + "\n")


def _task_for_document_type(document_type: str) -> str:
    if document_type in {"invoice", "receipt"}:
        return f"{document_type}_extraction"
    if document_type == "bank_statement":
        return "bank_statement_extraction"
    if document_type == "handwritten_note":
        return "handwritten_note_transcription"
    raise ValueError(f"unsupported document_type for metadata: {document_type}")


def build_record(
    kind: DocumentKind,
    profile: RegionProfile,
    rng: random.Random,
    index: int,
) -> dict[str, Any]:
    if kind in {"invoice", "receipt"}:
        return build_invoice_record(kind, profile, rng, index)
    if kind == "bank_statement":
        return build_bank_statement_record(profile, rng, index)
    if kind == "handwritten_note":
        return build_note_record(profile, rng, index)
    raise ValueError(f"unsupported document kind: {kind}")


def generate_documents(
    kind: DocumentKind,
    output_dir: Path,
    count: int,
    regions: tuple[str, ...] = SUPPORTED_REGIONS,
    seed: int = 20260607,
    augment: bool = True,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        region = regions[(index - 1) % len(regions)]
        profile = choose_profile(region, rng)
        record = build_record(kind, profile, rng, index)
        write_document_outputs(record, output_dir, rng, augment)
        records.append(record)
    append_metadata(output_dir, records)
    return records


def parse_args(kind: DocumentKind) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic" / kind)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--regions", nargs="+", default=list(SUPPORTED_REGIONS))
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def run_cli(kind: DocumentKind) -> None:
    args = parse_args(kind)
    records = generate_documents(
        kind=kind,
        output_dir=args.output,
        count=args.count,
        regions=tuple(args.regions),
        seed=args.seed,
        augment=not args.no_augment,
    )
    print(f"generated {len(records)} synthetic {kind} documents in {args.output}")
