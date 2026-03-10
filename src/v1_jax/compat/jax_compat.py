"""JAX compatibility patches for brainevent.

This module provides compatibility fixes for brainevent with JAX 0.9.x.
The issue is that brainevent uses ad.Zero.from_primal_value which was
removed in JAX 0.9.x.

Usage:
    from v1_jax.compat.jax_compat import apply_jax_compat_patches
    apply_jax_compat_patches()  # Call before importing brainevent
"""

import jax
import jax.core
import jax._src.ad_util as ad

_patches_applied = False


def _zero_from_primal_value(primal):
    """Create Zero from primal value - compatibility shim for JAX 0.9.x.

    This replaces the removed ad.Zero.from_primal_value method.

    Parameters
    ----------
    primal : array-like
        The primal value to create a Zero tangent for.

    Returns
    -------
    ad.Zero
        Zero tangent with the same abstract value as primal.
    """
    # Use jax.typeof for JAX 0.9.x (get_aval is deprecated)
    if hasattr(jax, 'typeof'):
        aval = jax.typeof(primal)
    else:
        aval = jax.core.get_aval(primal)
    return ad.Zero(aval)


def _zero_from_value(primal):
    """Create Zero from value - compatibility shim.

    Same as from_primal_value, kept for older JAX versions.
    """
    return _zero_from_primal_value(primal)


def apply_jax_compat_patches():
    """Apply compatibility patches for JAX 0.9.x.

    This function adds missing methods to ad.Zero that brainevent expects.
    Safe to call multiple times (idempotent).
    """
    global _patches_applied

    if _patches_applied:
        return

    # Check JAX version
    jax_version = tuple(int(x) for x in jax.__version__.split('.')[:3])

    # Add from_primal_value if missing (JAX 0.9.x)
    if not hasattr(ad.Zero, 'from_primal_value'):
        ad.Zero.from_primal_value = staticmethod(_zero_from_primal_value)

    # Add from_value if missing (for completeness)
    if not hasattr(ad.Zero, 'from_value'):
        ad.Zero.from_value = staticmethod(_zero_from_value)

    _patches_applied = True


def check_brainevent_compatibility():
    """Check if brainevent is compatible with current JAX version.

    Returns
    -------
    tuple
        (is_compatible: bool, message: str)
    """
    import jax

    jax_version = jax.__version__

    # Apply patches first
    apply_jax_compat_patches()

    # Try importing brainevent
    try:
        import brainevent
        return True, f"brainevent {brainevent.__version__} compatible with JAX {jax_version}"
    except Exception as e:
        return False, f"brainevent incompatible with JAX {jax_version}: {e}"
