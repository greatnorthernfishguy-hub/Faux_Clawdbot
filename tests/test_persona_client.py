from pathlib import Path

import pytest

import persona_client


EXPECTED_ROLES = {
    "strategist",
    "tracker",
    "wrench",
    "forge",
    "razor",
    "reviewer",
}


def test_complete_six_person_unit_is_packaged():
    assert persona_client.TQB_ROLES == EXPECTED_ROLES
    for role in EXPECTED_ROLES:
        assert (persona_client.PERSONALITY_DIR / f"{role}.md").is_file()
    assert persona_client.UNIT_DISCIPLINES_FILE.is_file()


@pytest.mark.parametrize(
    ("role", "identity_text"),
    [
        ("strategist", "Spec generation, decomposition, architecture planning"),
        ("tracker", "Bug hunting, root cause analysis, regression detection"),
        ("wrench", "Unconventional solutions, edge cases, creative problem-solving"),
        ("forge", "Integration, assembly, cross-module wiring"),
        ("razor", "Security review, attack surface analysis, policy enforcement"),
        ("reviewer", "Code quality, standards enforcement"),
    ],
)
def test_each_role_loads_its_own_canonical_identity(role, identity_text):
    personality, disciplines = persona_client.load_personality(role)

    assert identity_text in personality
    assert "Verify Before Asserting" in disciplines


def test_system_prompt_contains_persona_and_full_unit_disciplines():
    personality, disciplines = persona_client.load_personality("tracker")

    prompt = persona_client._build_system_prompt("tracker", personality, disciplines)

    assert "You are Tracker" in prompt
    assert "## 1. Verify Before Asserting" in prompt
    assert "## 2. The Graph Is Memory, Not Truth" in prompt
    assert "## 3. No Improvisation" in prompt
    assert "## 4. Report What You See, Not What You Expected" in prompt
    assert prompt.count("UNIT DISCIPLINES") == 0


def test_unknown_role_is_rejected_before_path_resolution():
    with pytest.raises(ValueError, match="Unknown TQB role"):
        persona_client.load_personality("../reviewer")


def test_role_specific_model_wins(monkeypatch):
    monkeypatch.setenv("CODEMINE_TRACKER_MODEL_ID", "provider/tracker-model")
    monkeypatch.setenv("CODEMINE_WORKER_MODEL_ID", "provider/shared-worker")
    monkeypatch.setenv("CODEMINE_RP_MODEL_ID", "provider/legacy")

    assert persona_client._get_rp_model_id("tracker") == "provider/tracker-model"
    assert persona_client._get_rp_model_id("forge") == "provider/shared-worker"


def test_legacy_model_setting_remains_a_fallback(monkeypatch):
    monkeypatch.delenv("CODEMINE_REVIEWER_MODEL_ID", raising=False)
    monkeypatch.delenv("CODEMINE_WORKER_MODEL_ID", raising=False)
    monkeypatch.setenv("CODEMINE_RP_MODEL_ID", "provider/legacy")

    assert persona_client._get_rp_model_id("reviewer") == "provider/legacy"


def test_docker_image_copies_the_persona_assets():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

    assert "ENV TQB_PERSONALITY_DIR=/app/tqb-personalities" in dockerfile
    assert "COPY tqb-personalities/ ./tqb-personalities/" in dockerfile
