"""Train and tune the OPTCG agent using RL Baselines3 Zoo.

This script registers the custom :class:`~rl.env.OPTCGEnv` environment and
extends the Zoo so that ``MaskablePPO`` can be used. It then delegates to
:func:`rl_zoo3.train.train`, so any CLI options supported by RL Baselines3 Zoo
are available.
"""

from gymnasium.envs.registration import register

# Register the environment with Gymnasium when this module is imported.
register(id="OPTCG-v0", entry_point="rl.env:OPTCGEnv")

from sb3_contrib import MaskablePPO
import rl_zoo3.utils as zoo_utils
import rl_zoo3.hyperparams_opt as hpo
from rl_zoo3.train import train

# Make MaskablePPO available in the Zoo CLI
zoo_utils.ALGOS["maskableppo"] = MaskablePPO
hpo.HYPERPARAMS_SAMPLER["maskableppo"] = hpo.sample_ppo_params


def main() -> None:
    """Launch the training process using RL Baselines3 Zoo."""
    train()


if __name__ == "__main__":  # pragma: no cover
    main()
