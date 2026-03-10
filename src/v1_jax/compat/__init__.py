"""Compatibility layer for JAX and brainevent.

This module provides compatibility fixes for JAX 0.9.x and brainevent.
Import this module BEFORE importing brainevent to apply necessary patches.

Usage:
    # At the top of your script, before any brainevent imports:
    from v1_jax.compat import apply_jax_compat_patches
    apply_jax_compat_patches()
    import brainevent  # Now safe to import
"""

from .jax_compat import apply_jax_compat_patches, check_brainevent_compatibility

__all__ = ['apply_jax_compat_patches', 'check_brainevent_compatibility']
