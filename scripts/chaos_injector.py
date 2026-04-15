"""Chaos scenario injector wrapper for Chaos Mesh manifests and shell injectors."""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    category: str
    kind: str
    file: str
    duration: str
    targets: list[str]
    description: str


class ChaosInjector:
    """Apply/delete Chaos Mesh scenarios using the central registry."""

    def __init__(
        self,
        namespace: str = "ewat",
        registry_path: str | Path = "k8s/chaos-mesh/registry.yaml",
        dry_run: bool = False,
    ) -> None:
        self._namespace = namespace
        self._dry_run = dry_run
        self._repo_root = Path(__file__).resolve().parents[1]
        self._registry_path = self._repo_root / Path(registry_path)
        self._scenarios = self._load_registry(self._registry_path)

    def list_scenarios(self) -> list[ScenarioSpec]:
        return sorted(self._scenarios.values(), key=lambda s: (s.category, s.name))

    def get_scenario(self, scenario_name: str) -> ScenarioSpec:
        try:
            return self._scenarios[scenario_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._scenarios))
            msg = f"Unknown scenario '{scenario_name}'. Available: {available}"
            raise KeyError(msg) from exc

    def apply(self, scenario_name: str) -> ScenarioSpec:
        spec = self.get_scenario(scenario_name)
        scenario_path = self._repo_root / "k8s/chaos-mesh" / spec.file
        self._ensure_exists(scenario_path)

        if scenario_path.suffix == ".yaml":
            self._run(["kubectl", "-n", self._namespace, "apply", "-f", str(scenario_path)])
        elif scenario_path.suffix == ".sh":
            self._run(["bash", str(scenario_path), "inject"])
        else:
            msg = f"Unsupported scenario file type: {scenario_path}"
            raise ValueError(msg)

        return spec

    def delete(self, scenario_name: str) -> ScenarioSpec:
        spec = self.get_scenario(scenario_name)
        scenario_path = self._repo_root / "k8s/chaos-mesh" / spec.file
        self._ensure_exists(scenario_path)

        if scenario_path.suffix == ".yaml":
            self._run(
                [
                    "kubectl",
                    "-n",
                    self._namespace,
                    "delete",
                    "-f",
                    str(scenario_path),
                    "--ignore-not-found=true",
                ]
            )
        elif scenario_path.suffix == ".sh":
            self._run(["bash", str(scenario_path), "cleanup"])
        else:
            msg = f"Unsupported scenario file type: {scenario_path}"
            raise ValueError(msg)

        return spec

    def _run(self, command: list[str]) -> None:
        if self._dry_run:
            logger.info("DRY-RUN: %s", " ".join(command))
            return

        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            logger.error("Command failed: %s", " ".join(command))
            logger.error("stdout: %s", proc.stdout.strip())
            logger.error("stderr: %s", proc.stderr.strip())
            raise RuntimeError(f"Command failed with code {proc.returncode}")

        if proc.stdout.strip():
            logger.info(proc.stdout.strip())

    @staticmethod
    def _load_registry(path: Path) -> dict[str, ScenarioSpec]:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        raw_scenarios: list[dict[str, Any]] = cfg.get("scenarios", [])

        scenarios: dict[str, ScenarioSpec] = {}
        for item in raw_scenarios:
            spec = ScenarioSpec(
                name=item["name"],
                category=item["category"],
                kind=item["kind"],
                file=item["file"],
                duration=item.get("duration", "0s"),
                targets=item.get("targets", []),
                description=item.get("description", ""),
            )
            scenarios[spec.name] = spec

        return scenarios

    @staticmethod
    def _ensure_exists(path: Path) -> None:
        if not path.exists():
            msg = f"Scenario file not found: {path}"
            raise FileNotFoundError(msg)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EWAT chaos scenario injector")
    parser.add_argument("action", choices=["list", "apply", "delete"])
    parser.add_argument("scenario", nargs="?", default="")
    parser.add_argument("--namespace", default="ewat")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    injector = ChaosInjector(namespace=args.namespace, dry_run=args.dry_run)

    if args.action == "list":
        for scenario in injector.list_scenarios():
            print(
                f"{scenario.name:24} {scenario.category:10} {scenario.kind:12} "
                f"{scenario.duration:>4} {scenario.file}"
            )
        return

    if not args.scenario:
        raise ValueError("scenario is required for apply/delete actions")

    if args.action == "apply":
        injector.apply(args.scenario)
    else:
        injector.delete(args.scenario)


if __name__ == "__main__":
    main()
