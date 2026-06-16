
from .adroit import AdroitEnv
from .dexart import DexArtEnv
from .metaworld import MetaWorldEnv, MetaWorldMultiViewEnv
from .dmc import make_dmc_env, DMCEnv, make_dmc_env_2d
from .ur5 import UR5Env

# Optional imports for Franka (requires zerorpc)
try:
    from .franka import FrankaEnv
    from .franka_pour import FrankaPourEnv
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import Franka environments: {e}. This is fine if you're not using Franka tasks.")
    FrankaEnv = None
    FrankaPourEnv = None

def __getattr__(name):
    if name == 'FlippingEnv':
        # Keep real-robot dependencies lazy so sim tasks importing rl_100.env do
        # not start flipping's keyboard listener in every eval worker.
        from .flipping import FlippingEnv
        return FlippingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

