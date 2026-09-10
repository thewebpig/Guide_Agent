"""Run the checked-in HFUT retrieval acceptance set without a model API."""

import json
from pathlib import Path

from guide_agent.app_settings import load_app_settings, resolve_product_path
from guide_agent.retrieval import build_scene_knowledge_index, search_knowledge


ROOT = Path(__file__).parents[1]


def main() -> int:
    settings = load_app_settings()
    threshold = settings.retrieval.minimum_score
    cases = json.loads((ROOT / "evals" / "hfut_retrieval.json").read_text(encoding="utf-8"))
    index = build_scene_knowledge_index(resolve_product_path(settings.scene.path))
    failures = 0
    for case in cases:
        results = search_knowledge(index, case["question"], 5)
        best = results[0]
        accepted = best["score"] >= threshold
        accepted_evidence = [item for item in results if item["score"] >= threshold]
        expected_text = case["expected"]
        passed = accepted == case["should_answer"] and (
            expected_text is None
            or any(expected_text in item["text"] for item in accepted_evidence)
        )
        failures += not passed
        print(
            f"{'PASS' if passed else 'FAIL'} score={best['score']:.3f} "
            f"accepted={accepted} question={case['question']}"
        )
    print(f"summary: {len(cases) - failures}/{len(cases)} passed; threshold={threshold:.2f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
