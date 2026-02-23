"""
Walrus-JAX: JAX/Flax translation of the Walrus PDE foundation model.

A 1-to-1 translation of the Walrus model architecture from PyTorch to JAX/Flax,
maintaining exact weight compatibility for pretrained checkpoint conversion.

Reference:
    Bodner et al., "Aurora: A Foundation Model of the Atmosphere" (2024)
    https://github.com/PolymathicAI/the_well
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_walrus")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_walrus.model import IsotropicModel

__all__ = ["__version__", "IsotropicModel"]
