"""The release gate in tools/audit.py: severity, the allowlist, and what is allowed to fail a release."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("vmd_audit", ROOT / "tools" / "audit.py")
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules["vmd_audit"] = audit
_SPEC.loader.exec_module(audit)


def test_allowlist_file_loads_and_every_entry_has_a_reason():
    rows = audit.load_allowlist(ROOT / "tools" / "audit-allowlist.toml")
    assert all(row.id and row.reason for row in rows)


def test_allowlist_rejects_an_entry_without_a_reason(tmp_path):
    path = tmp_path / "allow.toml"
    path.write_text('[[advisory]]\nid = "GHSA-aaaa-bbbb-cccc"\nreason = "  "\n', encoding="utf-8")
    with pytest.raises(ValueError, match="reason"):
        audit.load_allowlist(path)


def test_cvss_v3_critical_and_high_vectors():
    # CVSS:3.1 example, base score 9.8.
    assert audit.cvss_v3_base("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") == 9.8
    assert audit.score_band(9.8) == "critical"
    # PYSEC-2025-49 / CVE-2025-47273, the setuptools path traversal.
    setuptools = audit.cvss_v3_base("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H")
    assert setuptools is not None and setuptools >= 7.0
    assert audit.score_band(setuptools) == "high"


def test_reviewed_label_wins_over_a_high_cvss_vector():
    """GitHub calls the accelerate advisory moderate; the raw vector scores just over 7."""
    pysec = {
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:N/A:H"}],
    }
    ghsa = {"database_specific": {"severity": "MODERATE"}}

    def fetch(ident):
        return {"PYSEC-2026-3804": pysec, "GHSA-4j2p-28q2-5m79": ghsa}.get(ident)

    assert audit.lookup_severity(["PYSEC-2026-3804", "GHSA-4j2p-28q2-5m79"], fetch) == "medium"
    assert audit.severity_from_osv(pysec) == "high"
    # A secondary database saying "Medium" must not outrank GitHub's "LOW".
    labels = {"GHSA-rrmf-rvhw-rf47": {"database_specific": {"severity": "LOW"}},
              "BIT-pytorch-2025-3000": {"database_specific": {"severity": "Medium"}}}
    assert audit.lookup_severity(["BIT-pytorch-2025-3000", "GHSA-rrmf-rvhw-rf47"], labels.get) == "low"


def test_unknown_severity_is_blocking_and_an_allowlisted_high_is_not():
    payload = {
        "dependencies": [
            {"name": "setuptools", "version": "78.1.0", "vulns": [
                {"id": "PYSEC-2025-49", "aliases": ["CVE-2025-47273"], "fix_versions": ["78.1.1"]},
            ]},
            {"name": "mystery", "version": "1", "vulns": [
                {"id": "PYSEC-0000-1", "aliases": [], "fix_versions": []},
            ]},
        ],
    }

    def fetch(ident):
        if ident == "PYSEC-2025-49":
            return {"database_specific": {"severity": "HIGH"}}
        return None

    findings, skipped = audit.classify_pip(payload, [], fetch)
    assert skipped == []
    by_id = {item.ident: item for item in findings}
    assert by_id["PYSEC-2025-49"].blocks
    assert by_id["PYSEC-0000-1"].severity == "unknown" and by_id["PYSEC-0000-1"].blocks

    allow = [audit.Advisory("CVE-2025-47273", "patched in the next point release", "setuptools")]
    findings, _skipped = audit.classify_pip(payload, allow, fetch)
    accepted = next(item for item in findings if item.ident == "PYSEC-2025-49")
    assert accepted.accepted.startswith("patched") and not accepted.blocks


def test_duplicate_advisories_from_pip_audit_count_once():
    payload = {
        "dependencies": [
            {"name": "setuptools", "version": "78.1.0", "vulns": [
                {"id": "PYSEC-2025-49", "aliases": ["CVE-2025-47273"], "fix_versions": ["78.1.1"]},
                {"id": "PYSEC-2025-49", "aliases": ["CVE-2025-47273", "GHSA-5rjg-fvgr-3xxf"], "fix_versions": ["78.1.1"]},
            ]},
        ],
    }
    findings, _skipped = audit.classify_pip(payload, [], lambda _ident: {"database_specific": {"severity": "HIGH"}})
    assert len(findings) == 1


def test_medium_advisory_does_not_block_and_package_limit_is_honoured():
    payload = {
        "dependencies": [
            {"name": "accelerate", "version": "1.12.0", "vulns": [
                {"id": "PYSEC-2026-3804", "aliases": ["GHSA-4j2p-28q2-5m79"], "fix_versions": []},
            ]},
        ],
    }
    findings, _skipped = audit.classify_pip(payload, [], lambda _ident: {"database_specific": {"severity": "MODERATE"}})
    assert len(findings) == 1 and not findings[0].blocks
    other = audit.Advisory("PYSEC-2026-3804", "different package", "setuptools")
    assert audit.allowlist_reason(other, "accelerate", ["PYSEC-2026-3804"]) == ""


def test_cpu_torch_local_version_is_audited_as_the_pypi_release():
    assert audit._pypi_version("2.11.0+cpu") == "2.11.0"
    assert audit._pypi_version("2.11.0+cu128") == "2.11.0"
    assert audit._pypi_version("1.2.3") == "1.2.3"


def test_gitleaks_report_parses_a_list_and_an_empty_file():
    assert audit._parse_gitleaks("") == []
    rows = audit._parse_gitleaks('[{"File": "a.py", "StartLine": 3, "RuleID": "generic", "Description": "x"}]')
    assert rows[0]["RuleID"] == "generic"
