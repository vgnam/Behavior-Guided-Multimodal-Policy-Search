import inspect
from pathlib import Path
import unittest

import yaml

from runner.llm_num_optim_vision_runner import run_training_loop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class BMPSCMAGeneratedConfigTest(unittest.TestCase):
    def test_every_continuous_policy_bmps_config_has_valid_cma_variant(self):
        runner_parameters = set(inspect.signature(run_training_loop).parameters)
        sources = []
        for source in CONFIG_ROOT.rglob("*_bmps.yaml"):
            source_data = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
            if source_data.get("task") == "cont_state_llm_num_optim_vision":
                sources.append((source, source_data))

        self.assertGreater(len(sources), 0)
        for source, source_data in sources:
            target = source.with_name(
                source.name.removesuffix("_bmps.yaml") + "_bmps_cma.yaml"
            )
            self.assertTrue(target.exists(), f"Missing generated config for {source}")
            target_data = yaml.safe_load(target.read_text(encoding="utf-8-sig"))

            self.assertEqual(
                target_data["task"], "cont_state_llm_num_optim_vision_cma"
            )
            self.assertEqual(
                target_data["llm_si_template_name"], "num_optim_vision.j2"
            )
            self.assertEqual(
                target_data["llm_output_conversion_template_name"],
                "num_optim_vision.j2",
            )
            self.assertEqual(target_data["gym_env_name"], source_data["gym_env_name"])
            self.assertEqual(target_data["dim_actions"], source_data["dim_actions"])
            self.assertEqual(target_data["dim_states"], source_data["dim_states"])
            self.assertEqual(
                target_data.get("env_kwargs"), source_data.get("env_kwargs")
            )
            self.assertEqual(
                target_data.get("frame_sample_period"),
                source_data.get("frame_sample_period"),
            )
            self.assertFalse(target_data["stack_trajectory_frames"])

            unknown_keys = set(target_data) - runner_parameters
            self.assertFalse(
                unknown_keys,
                f"{target} contains keys not accepted by the runner: {unknown_keys}",
            )

    def test_generated_config_count_matches_continuous_source_count(self):
        source_count = sum(
            yaml.safe_load(path.read_text(encoding="utf-8-sig")).get("task")
            == "cont_state_llm_num_optim_vision"
            for path in CONFIG_ROOT.rglob("*_bmps.yaml")
        )
        target_count = len(list(CONFIG_ROOT.rglob("*_bmps_cma.yaml")))
        self.assertEqual(target_count, source_count)


if __name__ == "__main__":
    unittest.main()
