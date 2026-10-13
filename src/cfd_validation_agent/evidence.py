from __future__ import annotations

import hashlib

from .schemas import EvidenceItem, Severity


def evidence_id(source: str, line_number: int | None, category: str, message: str) -> str:
    value = f"{source}|{line_number}|{category}|{message}".encode("utf-8", errors="replace")
    return "ev-" + hashlib.sha256(value).hexdigest()[:12]


def make_evidence(
    *, source: str, category: str, severity: Severity, message: str,
    line_number: int | None = None, context: str = "", explicit: bool = True,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id(source, line_number, category, message),
        category=category, severity=severity, source=source,
        line_number=line_number, message=message, context=context, explicit=explicit,
    )
