"""[PHOT-107] Local-only Ollama swarm orchestration for inverse design."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_LOCAL_OLLAMA_MODEL
from .materials import BUILTIN_MATERIAL_CATALOG
from .specs import ProblemSpec, load_problem_spec, save_json
from .workflows import execute_design_cycle


def _extract_json_block(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Could not parse JSON from Ollama output: {text[:400]}")


def _clamp_int(value: Any, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp_float(value: Any, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sanitize_material_list(value: Any) -> list[str]:
    valid = {record.name for record in BUILTIN_MATERIAL_CATALOG if record.material_class in {"dielectric", "pcm"}}
    if not isinstance(value, list):
        return []
    deduped = []
    for item in value:
        name = str(item)
        if name in valid and name not in deduped:
            deduped.append(name)
    return deduped[:5]


ALLOWED_CHANGE_RULES = {
    "dataset.num_samples": lambda value: _clamp_int(value, 8, 64),
    "material_search_policy.max_candidates": lambda value: _clamp_int(value, 2, 6),
    "material_search_policy.include_materials": _sanitize_material_list,
    "surrogate.hidden_channels": lambda value: _clamp_int(value, 8, 128),
    "surrogate.epochs": lambda value: _clamp_int(value, 1, 12),
    "surrogate.batch_size": lambda value: _clamp_int(value, 2, 16),
    "surrogate.learning_rate": lambda value: _clamp_float(value, 1e-4, 5e-3),
    "surrogate.weight_decay": lambda value: _clamp_float(value, 0.0, 1e-2),
    "inverse_design.steps": lambda value: _clamp_int(value, 4, 128),
    "inverse_design.learning_rate": lambda value: _clamp_float(value, 0.01, 0.3),
    "inverse_design.temperature": lambda value: _clamp_float(value, 0.3, 2.0),
    "inverse_design.binarization_weight": lambda value: _clamp_float(value, 0.0, 0.2),
    "inverse_design.restarts": lambda value: _clamp_int(value, 1, 4),
    "inverse_design.top_k": lambda value: _clamp_int(value, 1, 5),
}


@dataclass(frozen=True)
class SwarmAgent:
    name: str
    mission: str
    allowed_fields: tuple[str, ...]


AGENTS = (
    SwarmAgent(
        name="photonics_strategist",
        mission="Adjust the design-space breadth and sampling budget without changing the physical target.",
        allowed_fields=(
            "dataset.num_samples",
            "material_search_policy.max_candidates",
            "material_search_policy.include_materials",
        ),
    ),
    SwarmAgent(
        name="surrogate_tuner",
        mission="Improve the forward surrogate's generalization and optimization stability.",
        allowed_fields=(
            "surrogate.hidden_channels",
            "surrogate.epochs",
            "surrogate.batch_size",
            "surrogate.learning_rate",
            "surrogate.weight_decay",
            "dataset.num_samples",
        ),
    ),
    SwarmAgent(
        name="inverse_tuner",
        mission="Improve inverse-search convergence and candidate quality without redefining the objective.",
        allowed_fields=(
            "inverse_design.steps",
            "inverse_design.learning_rate",
            "inverse_design.temperature",
            "inverse_design.binarization_weight",
            "inverse_design.restarts",
            "inverse_design.top_k",
        ),
    ),
    SwarmAgent(
        name="reviewer",
        mission="Review all agent proposals and merge only the highest-signal changes into one safe patch.",
        allowed_fields=tuple(ALLOWED_CHANGE_RULES.keys()),
    ),
)


class LocalOllamaClient:
    def __init__(self, binary: str | None = None, timeout_s: int = 120) -> None:
        self.binary = binary or self._resolve_binary()
        self.timeout_s = timeout_s

    def _resolve_binary(self) -> str:
        env_override = os.environ.get("OLLAMA_BIN")
        if env_override:
            return env_override
        direct = shutil.which("ollama")
        if direct:
            return direct
        candidate_paths = [
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            Path("/mnt/c/Users") / os.environ.get("USERNAME", "") / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        ]
        candidate_paths.extend(Path("/mnt/c/Users").glob("*/AppData/Local/Programs/Ollama/ollama.exe"))
        for path in candidate_paths:
            if path and Path(path).exists():
                return str(path)
        raise FileNotFoundError("Could not find a local Ollama binary. Set OLLAMA_BIN if needed.")

    def list_models(self) -> list[dict[str, str]]:
        completed = subprocess.run(
            [self.binary, "list"],
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
        )
        models = []
        for line in completed.stdout.splitlines()[1:]:
            if not line.strip():
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 4:
                continue
            models.append({"name": parts[0], "id": parts[1], "size": parts[2], "modified": parts[3]})
        return models

    def ensure_local_model(self, model: str) -> None:
        if "cloud" in model.lower():
            raise ValueError(f"Model {model!r} is cloud-backed and not allowed for this local-only swarm.")
        local_models = {row["name"]: row for row in self.list_models()}
        if model not in local_models:
            raise ValueError(f"Model {model!r} is not installed in the local Ollama runtime.")
        if local_models[model]["size"] == "-":
            raise ValueError(f"Model {model!r} is not stored locally and is not allowed.")

    def generate_json(self, model: str, prompt: str) -> dict[str, Any]:
        self.ensure_local_model(model)
        completed = subprocess.run(
            [self.binary, "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            timeout=self.timeout_s,
        )
        return _extract_json_block(completed.stdout.strip())


class ScriptedOllamaClient:
    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {name: list(items) for name, items in responses.items()}

    def ensure_local_model(self, model: str) -> None:
        return

    def generate_json(self, model: str, prompt: str) -> dict[str, Any]:
        match = re.search(r"You are agent ([a-z_]+)", prompt)
        agent_name = match.group(1) if match else "reviewer"
        if agent_name not in self.responses or not self.responses[agent_name]:
            raise ValueError(f"No scripted response available for agent {agent_name!r}.")
        return self.responses[agent_name].pop(0)


def sanitize_changes(changes: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in changes.items():
        if key not in ALLOWED_CHANGE_RULES:
            continue
        normalized = ALLOWED_CHANGE_RULES[key](value)
        if normalized not in (None, [], {}):
            sanitized[key] = normalized
    return sanitized


def apply_changes(spec_dict: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(spec_dict)
    for dotted_key, value in changes.items():
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return updated


def build_agent_prompt(
    agent: SwarmAgent,
    spec_dict: dict[str, Any],
    latest_summary: dict[str, Any],
    previous_rounds: list[dict[str, Any]],
    proposals: list[dict[str, Any]] | None = None,
) -> str:
    allowed = "\n".join(f"- {field}" for field in agent.allowed_fields)
    payload = {
        "problem_id": spec_dict["problem_id"],
        "solver_backend": spec_dict["solver"]["backend"],
        "grid_shape": spec_dict["geometry"]["grid_shape"],
        "materials": spec_dict["material_search_policy"].get("include_materials", []),
        "dataset": spec_dict["dataset"],
        "surrogate": spec_dict["surrogate"],
        "inverse_design": spec_dict["inverse_design"],
        "latest_summary": latest_summary,
        "history": previous_rounds[-2:],
    }
    reviewer_context = ""
    if proposals:
        reviewer_context = f"\nAgent proposals:\n{json.dumps(proposals, indent=2)}\n"
    return (
        f"You are agent {agent.name}.\n"
        f"Mission: {agent.mission}\n"
        "Context: this is a local-only Ollama swarm for a freeform periodic metasurface inverse-design problem.\n"
        "Do not use Hugging Face, cloud services, or remote models. Do not change the physical target.\n"
        f"Allowed dotted keys:\n{allowed}\n"
        f"{reviewer_context}"
        f"Current state:\n{json.dumps(payload, indent=2)}\n"
        "Return strict JSON only with shape:\n"
        '{"summary": "short rationale", "changes": {"dotted.key": value}}\n'
        "Propose at most 4 changes.\n"
    )


def build_iteration_spec(
    base_spec_dict: dict[str, Any],
    base_problem_id: str,
    round_index: int,
) -> dict[str, Any]:
    updated = deepcopy(base_spec_dict)
    updated["problem_id"] = f"{base_problem_id}_swarm_r{round_index:02d}"
    return updated


def run_swarm(
    spec_path: str | Path,
    model: str = DEFAULT_LOCAL_OLLAMA_MODEL,
    rounds: int = 1,
    client: LocalOllamaClient | ScriptedOllamaClient | None = None,
    refine: bool = True,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    spec_source = Path(spec_path).resolve()
    base_spec = load_problem_spec(spec_source)
    base_spec_dict = base_spec.to_dict()
    swarm_dir = base_spec.output_dir / base_spec.problem_id / "swarm"
    specs_dir = swarm_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    active_client = client or LocalOllamaClient()
    active_client.ensure_local_model(model)

    history: list[dict[str, Any]] = []
    current_spec_dict = deepcopy(base_spec_dict)

    for round_index in range(rounds + 1):
        if round_index == 0:
            changes = {}
            agent_entries = []
        else:
            latest_summary = history[-1]["summary"]
            proposals = []
            agent_entries = []
            for agent in AGENTS[:-1]:
                prompt = build_agent_prompt(agent, current_spec_dict, latest_summary, history)
                raw_response = active_client.generate_json(model, prompt)
                sanitized = sanitize_changes(raw_response.get("changes", {}))
                agent_entries.append(
                    {
                        "name": agent.name,
                        "prompt": prompt,
                        "response": raw_response,
                        "sanitized_changes": sanitized,
                    }
                )
                proposals.append(
                    {
                        "agent": agent.name,
                        "summary": raw_response.get("summary", ""),
                        "changes": sanitized,
                    }
                )
            reviewer = AGENTS[-1]
            reviewer_prompt = build_agent_prompt(reviewer, current_spec_dict, latest_summary, history, proposals=proposals)
            reviewer_response = active_client.generate_json(model, reviewer_prompt)
            changes = sanitize_changes(reviewer_response.get("changes", {}))
            if not changes:
                for proposal in proposals:
                    if proposal["changes"]:
                        changes = proposal["changes"]
                        break
            agent_entries.append(
                {
                    "name": reviewer.name,
                    "prompt": reviewer_prompt,
                    "response": reviewer_response,
                    "sanitized_changes": changes,
                }
            )
            current_spec_dict = apply_changes(current_spec_dict, changes)

        iteration_spec_dict = build_iteration_spec(current_spec_dict, base_spec.problem_id, round_index)
        iteration_spec_path = specs_dir / f"round_{round_index:02d}.json"
        save_json(iteration_spec_path, iteration_spec_dict)
        iteration_spec = load_problem_spec(iteration_spec_path)
        summary = execute_design_cycle(iteration_spec, refine=refine)
        history.append(
            {
                "round": round_index,
                "spec_path": str(iteration_spec_path),
                "changes": changes,
                "summary": summary,
                "agents": agent_entries,
            }
        )

    best_round = min(history, key=lambda item: item["summary"]["best_candidate_loss"])
    payload = {
        "model": model,
        "base_spec": str(spec_source),
        "rounds": rounds,
        "best_round": best_round["round"],
        "history": history,
    }
    save_json(swarm_dir / "swarm_log.json", payload)
    return payload
