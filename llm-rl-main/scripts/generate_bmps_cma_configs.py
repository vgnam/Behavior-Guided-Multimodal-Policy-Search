"""Generate BMPS-CMA configs from every continuous-policy BMPS config.

The source configs remain authoritative for environment dimensions, models,
horizons, environment kwargs, and API overrides. Run this script after adding or
changing a ``cont_state_llm_num_optim_vision`` BMPS configuration.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"

SOURCE_TASK = "cont_state_llm_num_optim_vision"
TARGET_TASK = "cont_state_llm_num_optim_vision_cma"

OBSOLETE_BMPS_KEYS = {
    "n_neighbors",
    "poisson_lam",
    "neighbor_step",
    "ablate_anchor",
}

API_KEYS = (
    "llm_api_key",
    "llm_api_base",
    "vlm_api_key",
    "vlm_api_base",
)

BMPS_CMA_DEFAULTS = {
    "candidate_evaluation_episodes": 1,
    "pairwise_max_comparisons": 4,
    "stack_trajectory_frames": False,
    "stack_motion_threshold": 18.0,
    "stack_background_learning_rate": 0.01,
    "stack_tint_strength": 0.45,
    "stack_occupancy_scale": 4.0,
    "cma_population_size": 13,
    "cma_sigma": 0.5,
    "cma_covariance_mode": "auto",
    "cma_full_covariance_max_dim": 128,
    "cma_lower_bound": -6.0,
    "cma_upper_bound": 6.0,
    "trust_region_radius": 3.0,
    "llm_guidance_alpha": 1.0,
    "cma_acceptance_margin": 0.0,
    "cma_restart_distance": 8.0,
    "cma_restart_margin": 0.0,
    "cma_seed": 0,
}


def bmps_cma_logdir(logdir: str) -> str:
    updated, substitutions = re.subn(
        r"bmps(?!_cma)", "bmps_cma", str(logdir), count=1
    )
    return updated if substitutions else f"{logdir}_bmps_cma"


def generate_config(source: Path) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if data.get("task") != SOURCE_TASK:
        raise ValueError(f"{source} is not a continuous-policy BMPS config")

    generated = {}
    for key, value in data.items():
        if key in OBSOLETE_BMPS_KEYS or key in API_KEYS:
            continue
        if key == "task":
            value = TARGET_TASK
        elif key == "logdir":
            value = bmps_cma_logdir(value)
        elif key in {"llm_si_template_name", "llm_output_conversion_template_name"}:
            value = "num_optim_vision.j2"
        generated[key] = value

    generated.update(BMPS_CMA_DEFAULTS)
    for key in API_KEYS:
        if key in data:
            generated[key] = data[key]

    destination = source.with_name(
        source.name.removesuffix("_bmps.yaml") + "_bmps_cma.yaml"
    )
    header = (
        "# Generated from "
        f"{source.name} by scripts/generate_bmps_cma_configs.py.\n"
        "# Environment/model settings follow the source BMPS configuration.\n"
    )
    destination.write_text(
        header + yaml.safe_dump(generated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    sources = sorted(CONFIG_ROOT.rglob("*_bmps.yaml"))
    continuous_sources = []
    for source in sources:
        data = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        if data.get("task") == SOURCE_TASK:
            continuous_sources.append(source)

    generated = [generate_config(source) for source in continuous_sources]
    for path in generated:
        print(path.relative_to(PROJECT_ROOT))
    print(f"Generated {len(generated)} BMPS-CMA configurations.")


if __name__ == "__main__":
    main()
