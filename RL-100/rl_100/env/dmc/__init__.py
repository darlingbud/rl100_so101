# from .dmc import make_dmc_Env
from .dmc import make_dmc_env, DMCEnv
from .dmc_timestep import make_dmc_env_2d
try:
    from .dm_control_utils import register_dmc_envs

    register_dmc_envs()
except:
    pass