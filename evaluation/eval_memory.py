"""Live multi-session episodic and Semantic memory benchmark.

This runner exercises the public API against a running MindBridge stack. It
uses disposable users, real chat turns, the outbox worker, and LangGraph Store;
it is an evaluation artifact rather than a pytest suite.

Usage:
    cd evaluation
    uv run --project ../backend python eval_memory.py
    uv run --project ../backend python eval_memory.py --max-recall 1 --skip-gates
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

EVALUATION_ERRORS = (httpx.HTTPError, TimeoutError, KeyError, TypeError, ValueError)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EVAL_DIR / "configs" / "memory_eval.yaml"
DATASET_PATH = EVAL_DIR / "datasets" / "memory_benchmark.json"
RESULTS_PATH = EVAL_DIR / "results" / "memory_eval_results.csv"
SUMMARY_PATH = EVAL_DIR / "results" / "memory_eval_summary.csv"

load_dotenv(ROOT / "backend" / ".env")

_registration_lock = asyncio.Lock()
_next_registration_at = 0.0
_deletion_lock = asyncio.Lock()
_next_deletion_at = 0.0

RESULT_FIELDS = [
    "case_id",
    "kind",
    "condition",
    "passed",
    "recalled",
    "grounded",
    "item_count",
    "source_latency_s",
    "probe_latency_s",
    "response_excerpt",
    "details",
    "error",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def matches_all_groups(text: str, groups: list[list[str]]) -> bool:
    normalized = text.casefold()
    return all(any(term.casefold() in normalized for term in group) for group in groups)


def flatten_items(items: list[dict[str, Any]]) -> str:
    return json.dumps(items, ensure_ascii=False, sort_keys=True)


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


async def pace_registration(interval_s: float) -> None:
    """Keep the live benchmark within the production auth rate limit."""
    global _next_registration_at
    async with _registration_lock:
        delay = _next_registration_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        _next_registration_at = time.monotonic() + interval_s


async def pace_deletion(interval_s: float) -> None:
    """Keep disposable-account cleanup within the auth deletion limit."""
    global _next_deletion_at
    async with _deletion_lock:
        delay = _next_deletion_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        _next_deletion_at = time.monotonic() + interval_s


class LiveUser:
    def __init__(self, cfg: dict[str, Any], label: str):
        self.cfg = cfg
        self.password = "MindBridgeMemoryEval2026!"
        suffix = uuid.uuid4().hex[:12]
        self.email = f"memory-eval-{label}-{suffix}@example.com"
        self.client = httpx.AsyncClient(
            base_url=cfg["backend_url"],
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        self.token = ""

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def create(self) -> None:
        api_key_env = self.cfg["provider_api_key_env"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} is required for the live memory evaluation."
            )
        interval_s = float(self.cfg["registration_interval_s"])
        for _ in range(6):
            await pace_registration(interval_s)
            response = await self.client.post(
                "/auth/register",
                json={
                    "email": self.email,
                    "password": self.password,
                    "display_name": "Memory Eval",
                },
            )
            if response.status_code != 429:
                break
        response.raise_for_status()
        self.token = response.json()["access_token"]
        response = await self.client.put(
            "/api-keys",
            headers=self.headers,
            json={
                "provider": self.cfg["provider"],
                "api_key": api_key,
                "base_url": self.cfg["provider_base_url"],
                "model_id": self.cfg["model"],
                "display_name": "Memory Eval Model",
            },
        )
        response.raise_for_status()

    async def set_memory(self, enabled: bool) -> dict[str, Any]:
        response = await self.client.patch(
            "/memory",
            headers=self.headers,
            json={"memory_enabled": enabled},
        )
        response.raise_for_status()
        return response.json()

    async def memory(self, category: str = "episodes") -> dict[str, Any]:
        response = await self.client.get(
            f"/memory?category={category}&limit=100",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    async def clear_memory(self) -> dict[str, Any]:
        response = await self.client.delete("/memory", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
    ) -> tuple[str, float, str]:
        started = time.perf_counter()
        response = await self.client.post(
            "/chat",
            headers=self.headers,
            json={
                "message": message,
                "conversation_id": conversation_id,
                "model": self.cfg["model"],
                "provider": self.cfg["provider"],
                "stream": False,
            },
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        payload = response.json()
        return (
            payload["message"]["content"],
            round(elapsed, 3),
            payload["conversation_id"],
        )

    async def wait_for_items(
        self,
        minimum: int = 1,
        *,
        category: str = "episodes",
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.cfg["memory_wait_timeout_s"]
        while time.monotonic() < deadline:
            snapshot = await self.memory(category)
            if len(snapshot["items"]) >= minimum:
                return snapshot
            await asyncio.sleep(self.cfg["poll_interval_s"])
        raise TimeoutError(
            f"Memory writer did not produce {minimum} {category} item(s) in time."
        )

    async def wait_for_episode_revision(
        self, conversation_id: str, minimum: int
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.cfg["memory_wait_timeout_s"]
        while time.monotonic() < deadline:
            snapshot = await self.memory("episodes")
            for item in snapshot["items"]:
                if (
                    item["key"] == conversation_id
                    and int(item["value"].get("target_revision", 0)) >= minimum
                ):
                    return snapshot
            await asyncio.sleep(self.cfg["poll_interval_s"])
        raise TimeoutError(
            f"Episode {conversation_id} did not reach revision {minimum} in time."
        )

    async def _delete_account(self) -> httpx.Response:
        interval_s = float(self.cfg["account_deletion_interval_s"])
        for _ in range(6):
            await pace_deletion(interval_s)
            response = await self.client.request(
                "DELETE",
                "/auth/me",
                headers=self.headers,
                json={"password": self.password},
            )
            if response.status_code != 429:
                return response
        return response

    async def delete_account_and_verify(self) -> tuple[bool, bool]:
        old_headers = self.headers
        response = await self._delete_account()
        response.raise_for_status()
        self.token = ""
        rejected = await self.client.get("/memory", headers=old_headers)
        login = await self.client.post(
            "/auth/login",
            json={"email": self.email, "password": self.password},
        )
        return rejected.status_code == 401, login.status_code == 401

    async def delete(self) -> None:
        try:
            if self.token:
                response = await self._delete_account()
                response.raise_for_status()
        finally:
            await self.client.aclose()


async def cleanup(*users: LiveUser | None) -> None:
    for user in users:
        if user is None:
            continue
        try:
            await user.delete()
        except EVALUATION_ERRORS as exc:
            print(f"  cleanup warning for {user.email}: {exc}")


async def run_recall_case(
    cfg: dict[str, Any],
    scenario: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    user = LiveUser(cfg, f"{scenario['id']}-{condition}")
    row = {
        "case_id": scenario["id"],
        "kind": "paired_recall",
        "condition": condition,
        "passed": False,
        "recalled": False,
        "grounded": False,
        "item_count": 0,
        "source_latency_s": 0.0,
        "probe_latency_s": 0.0,
        "response_excerpt": "",
        "details": "",
        "error": "",
    }
    try:
        await user.create()
        if condition == "on":
            await user.set_memory(True)
        _, row["source_latency_s"], _ = await user.chat(scenario["source"])
        snapshot = (
            await user.wait_for_items() if condition == "on" else await user.memory()
        )
        row["item_count"] = len(snapshot["items"])
        row["grounded"] = matches_all_groups(
            flatten_items(snapshot["items"]),
            scenario["expected_groups"],
        )
        response, row["probe_latency_s"], _ = await user.chat(scenario["probe"])
        row["response_excerpt"] = response[:500].strip()
        row["recalled"] = matches_all_groups(response, scenario["expected_groups"])
        row["passed"] = (
            row["recalled"] and row["grounded"]
            if condition == "on"
            else not row["recalled"] and row["item_count"] == 0
        )
    except EVALUATION_ERRORS as exc:
        row["error"] = str(exc)
    finally:
        await cleanup(user)
    return row


async def prepare_memory(
    user: LiveUser,
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], bool, float]:
    await user.create()
    await user.set_memory(True)
    _, source_latency, _ = await user.chat(scenario["source"])
    snapshot = await user.wait_for_items()
    grounded = matches_all_groups(
        flatten_items(snapshot["items"]),
        scenario["expected_groups"],
    )
    return snapshot, grounded, source_latency


async def run_semantic_case(
    cfg: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    user = LiveUser(cfg, scenario["id"])
    row = gate_row({"id": scenario["id"], "kind": "semantic_recall"})
    row["condition"] = scenario["semantic_kind"]
    try:
        await user.create()
        await user.set_memory(True)
        _, row["source_latency_s"], _ = await user.chat(scenario["source"])
        snapshot = await user.wait_for_items(category="semantic")
        expected_items = [
            item
            for item in snapshot["items"]
            if item["value"].get("kind") == scenario["semantic_kind"]
        ]
        row["item_count"] = len(snapshot["items"])
        row["grounded"] = bool(expected_items) and matches_all_groups(
            flatten_items(expected_items),
            scenario["expected_groups"],
        )
        response, row["probe_latency_s"], _ = await user.chat(scenario["probe"])
        row["response_excerpt"] = response[:500].strip()
        row["recalled"] = matches_all_groups(response, scenario["expected_groups"])
        row["passed"] = row["grounded"] and row["recalled"]
        row["details"] = (
            f"Expected one grounded {scenario['semantic_kind']} fact and "
            "cross-session recall."
        )
    except EVALUATION_ERRORS as exc:
        row["error"] = str(exc)
    finally:
        await cleanup(user)
    return row


def gate_row(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": scenario["id"],
        "kind": scenario["kind"],
        "condition": "gate",
        "passed": False,
        "recalled": False,
        "grounded": False,
        "item_count": 0,
        "source_latency_s": 0.0,
        "probe_latency_s": 0.0,
        "response_excerpt": "",
        "details": "",
        "error": "",
    }


async def run_gate_case(
    cfg: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    kind = scenario["kind"]
    row = gate_row(scenario)
    primary: LiveUser | None = None
    secondary: LiveUser | None = None
    try:
        if kind == "crisis_exclusion":
            primary = LiveUser(cfg, scenario["id"])
            await primary.create()
            await primary.set_memory(True)
            _, row["source_latency_s"], _ = await primary.chat(scenario["source"])
            await asyncio.sleep(5)
            snapshot = await primary.memory()
            semantic = await primary.memory("semantic")
            row["item_count"] = len(snapshot["items"])
            row["passed"] = row["item_count"] == 0 and not semantic["items"]
            row["details"] = "Crisis turns must not write episodic or Semantic memory."
            return row

        if kind == "isolation":
            primary = LiveUser(cfg, f"{scenario['id']}-owner")
            snapshot, grounded, source_latency = await prepare_memory(primary, scenario)
            secondary = LiveUser(cfg, f"{scenario['id']}-other")
            await secondary.create()
            await secondary.set_memory(True)
            other_snapshot = await secondary.memory()
            response, probe_latency, _ = await secondary.chat(scenario["probe"])
            row.update(
                grounded=grounded,
                item_count=len(other_snapshot["items"]),
                source_latency_s=source_latency,
                probe_latency_s=probe_latency,
                response_excerpt=response[:500].strip(),
                recalled=matches_all_groups(response, scenario["expected_groups"]),
            )
            row["passed"] = (
                grounded
                and len(snapshot["items"]) >= 1
                and row["item_count"] == 0
                and not row["recalled"]
            )
            row["details"] = (
                "A second user must not list or recall the owner's episode."
            )
            return row

        if kind == "semantic_rejection":
            primary = LiveUser(cfg, scenario["id"])
            await primary.create()
            await primary.set_memory(True)
            _, row["source_latency_s"], _ = await primary.chat(scenario["source"])
            await primary.wait_for_items(category="episodes")
            semantic = await primary.memory("semantic")
            row["item_count"] = len(semantic["items"])
            row["passed"] = row["item_count"] == 0
            row["details"] = (
                "A non-explicit statement must not be promoted to Semantic memory."
            )
            return row

        if kind == "idempotency":
            primary = LiveUser(cfg, scenario["id"])
            await primary.create()
            await primary.set_memory(True)
            _, row["source_latency_s"], conversation_id = await primary.chat(
                scenario["source"]
            )
            first = await primary.wait_for_items(category="semantic")
            first_items = first["items"]
            if len(first_items) != 1:
                raise ValueError(
                    "Expected exactly one Semantic item after the first write."
                )
            key = first_items[0]["key"]
            _, row["probe_latency_s"], _ = await primary.chat(
                scenario["source"],
                conversation_id=conversation_id,
            )
            episodes = await primary.wait_for_episode_revision(conversation_id, 2)
            second = await primary.memory("semantic")
            row["item_count"] = len(second["items"]) + len(episodes["items"])
            row["grounded"] = matches_all_groups(
                flatten_items(second["items"]), scenario["expected_groups"]
            )
            row["passed"] = (
                row["grounded"]
                and len(second["items"]) == 1
                and len(episodes["items"]) == 1
                and second["items"][0]["key"] == key
                and int(second["items"][0]["value"].get("version", 0)) == 1
            )
            row["details"] = (
                "Repeating one explicit fact in the same conversation must keep "
                "one Semantic version and advance one Episode, not duplicate either."
            )
            return row

        primary = LiveUser(cfg, scenario["id"])
        snapshot, grounded, source_latency = await prepare_memory(primary, scenario)
        row.update(
            grounded=grounded,
            item_count=len(snapshot["items"]),
            source_latency_s=source_latency,
        )

        if kind == "consent_off":
            status = await primary.set_memory(False)
            retained = await primary.memory()
            response, probe_latency, _ = await primary.chat(scenario["probe"])
            row.update(
                item_count=len(retained["items"]),
                probe_latency_s=probe_latency,
                response_excerpt=response[:500].strip(),
                recalled=matches_all_groups(response, scenario["expected_groups"]),
            )
            row["passed"] = (
                grounded
                and not status["memory_enabled"]
                and row["item_count"] >= 1
                and not row["recalled"]
            )
            row["details"] = (
                "Consent off retains inspectable data but blocks Selection and writes."
            )
        elif kind == "clear":
            deleted = await primary.clear_memory()
            cleared = await primary.memory()
            response, probe_latency, _ = await primary.chat(scenario["probe"])
            row.update(
                item_count=len(cleared["items"]),
                probe_latency_s=probe_latency,
                response_excerpt=response[:500].strip(),
                recalled=matches_all_groups(response, scenario["expected_groups"]),
            )
            row["passed"] = (
                grounded
                and deleted["deleted_items"] >= 1
                and row["item_count"] == 0
                and not row["recalled"]
            )
            row["details"] = (
                "Clear must disable memory, delete Store items, and block recall."
            )
        elif kind == "irrelevant_rejection":
            response, probe_latency, _ = await primary.chat(scenario["probe"])
            row.update(
                probe_latency_s=probe_latency,
                response_excerpt=response[:500].strip(),
                recalled=matches_all_groups(response, scenario["expected_groups"]),
            )
            row["passed"] = grounded and not row["recalled"]
            row["details"] = "An unrelated turn must not surface the stored canary."
        elif kind == "account_deletion":
            token_rejected, login_rejected = await primary.delete_account_and_verify()
            row["passed"] = grounded and token_rejected and login_rejected
            row["details"] = (
                "Account deletion must invalidate the access token and remove login identity."
            )
        else:
            raise ValueError(f"Unknown gate kind: {kind}")
    except EVALUATION_ERRORS as exc:
        row["error"] = str(exc)
    finally:
        await cleanup(secondary, primary)
    return row


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(bool(row[field]) for row in rows) / len(rows), 4) if rows else 0.0


def write_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = [row for row in rows if row["kind"] == "paired_recall"]
    recall_on = [row for row in paired if row["condition"] == "on"]
    recall_off = [row for row in paired if row["condition"] == "off"]
    semantic = [row for row in rows if row["kind"] == "semantic_recall"]
    gates = [row for row in rows if row["condition"] == "gate"]
    on_rate = rate(recall_on, "recalled")
    off_rate = rate(recall_off, "recalled")
    summary = [
        {"metric": "recall_on_rate", "n": len(recall_on), "value": on_rate},
        {"metric": "recall_off_rate", "n": len(recall_off), "value": off_rate},
        {
            "metric": "recall_lift",
            "n": len(recall_on),
            "value": round(on_rate - off_rate, 4),
        },
        {
            "metric": "grounded_storage_rate",
            "n": len(recall_on),
            "value": rate(recall_on, "grounded"),
        },
        {
            "metric": "paired_expected_behavior_pass_rate",
            "n": len(paired),
            "value": rate(paired, "passed"),
        },
        {
            "metric": "semantic_storage_rate",
            "n": len(semantic),
            "value": rate(semantic, "grounded"),
        },
        {
            "metric": "semantic_recall_rate",
            "n": len(semantic),
            "value": rate(semantic, "recalled"),
        },
        {
            "metric": "semantic_expected_behavior_pass_rate",
            "n": len(semantic),
            "value": rate(semantic, "passed"),
        },
        {
            "metric": "privacy_safety_idempotency_gate_pass_rate",
            "n": len(gates),
            "value": rate(gates, "passed"),
        },
        {
            "metric": "overall_expected_behavior_pass_rate",
            "n": len(rows),
            "value": rate(rows, "passed"),
        },
        {
            "metric": "execution_error_rate",
            "n": len(rows),
            "value": rate(rows, "error"),
        },
    ]
    with SUMMARY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "n", "value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)
    return summary


async def run(
    max_recall: int | None,
    skip_gates: bool,
    skip_semantic: bool,
) -> None:
    cfg = load_yaml(CONFIG_PATH)
    dataset = load_json(DATASET_PATH)
    recall_scenarios = dataset["recall_scenarios"]
    if max_recall is not None:
        recall_scenarios = recall_scenarios[:max_recall]

    total = len(recall_scenarios) * 2
    if not skip_semantic:
        total += len(dataset["semantic_scenarios"])
    if not skip_gates:
        total += len(dataset["gate_scenarios"])
    print(
        f"Memory benchmark: {total} cases using {cfg['model']} "
        f"and {cfg['embedding_model']}"
    )

    rows: list[dict[str, Any]] = []
    for scenario in recall_scenarios:
        for condition in ("off", "on"):
            print(f"  {scenario['id']} [{condition}] ...", end=" ", flush=True)
            row = await run_recall_case(cfg, scenario, condition)
            rows.append(row)
            write_results(rows)
            print("PASS" if row["passed"] else f"FAIL ({row['error'] or 'behavior'})")

    if not skip_semantic:
        for scenario in dataset["semantic_scenarios"]:
            print(
                f"  {scenario['id']} [{scenario['semantic_kind']}] ...",
                end=" ",
                flush=True,
            )
            row = await run_semantic_case(cfg, scenario)
            rows.append(row)
            write_results(rows)
            print("PASS" if row["passed"] else f"FAIL ({row['error'] or 'behavior'})")

    if not skip_gates:
        for scenario in dataset["gate_scenarios"]:
            print(f"  {scenario['id']} ...", end=" ", flush=True)
            row = await run_gate_case(cfg, scenario)
            rows.append(row)
            write_results(rows)
            print("PASS" if row["passed"] else f"FAIL ({row['error'] or 'behavior'})")

    summary = write_summary(rows)
    print("\nSummary")
    for item in summary:
        print(f"  {item['metric']}: {item['value']} (n={item['n']})")
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live memory on/off benchmark")
    parser.add_argument("--max-recall", type=int, default=None)
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.max_recall, args.skip_gates, args.skip_semantic))


if __name__ == "__main__":
    main()
