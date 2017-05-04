from rllab.algos.ddpg_unified import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


batch_size_value = 64

number_of_episodes = 3000

from rllab.envs.mujoco.hopper_env import HopperEnv

env = normalize(HopperEnv())


def run_task(*_):
    env = normalize(HopperEnv())

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 100)
    )

    es = OUStrategy(env_spec=env.spec)

    qf = ContinuousMLPQFunction(
        env_spec=env.spec
        hidden_sizes =(100,100)
    )

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=batch_size_value,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=number_of_episodes,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=0.001,
        policy_learning_rate=0.0001,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_name="Unified_Policy_Gradients/" + "Unified_DDPG_Hopper",
    seed=1,
    # plot=True,
)
