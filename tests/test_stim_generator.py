"""Tests for stimulus generator module.

Tests cover:
- Drifting grating generation
- Stimulus configuration validation
- Firing rate conversion
- Classification label generation
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from v1_jax.data.stim_generator import (
    StimulusConfig,
    make_drifting_grating,
    generate_grating_with_delays,
    firing_rates_to_input,
    create_classification_labels,
    create_image_labels,
    prepare_image_for_lgn,
    create_stimulus_sequence,
    compute_n_chunks,
    validate_config,
)


class TestDriftingGrating:
    """Tests for drifting grating stimulus generation."""

    def test_grating_shape_static(self):
        """Test static grating output shape."""
        grating = make_drifting_grating(
            row_size=120,
            col_size=240,
            moving_flag=False,
            image_duration=100,
        )
        # Output shape depends on spatial sampling (cpd-based)
        # Just check basic structure
        assert grating.ndim == 4
        assert grating.shape[0] == 100
        assert grating.shape[-1] == 1

    def test_grating_shape_moving(self):
        """Test moving grating output shape."""
        grating = make_drifting_grating(
            row_size=60,
            col_size=120,
            moving_flag=True,
            image_duration=50,
        )
        assert grating.ndim == 4
        assert grating.shape[0] == 50
        assert grating.shape[-1] == 1

    def test_grating_value_range(self):
        """Test grating values are in expected range."""
        grating = make_drifting_grating(
            contrast=1.0,
            image_duration=10,
        )
        assert grating.min() >= -1.0
        assert grating.max() <= 1.0

    def test_grating_contrast(self):
        """Test contrast affects amplitude."""
        grating_full = make_drifting_grating(contrast=1.0, image_duration=10)
        grating_half = make_drifting_grating(contrast=0.5, image_duration=10)

        assert abs(grating_full).max() > abs(grating_half).max()

    def test_grating_with_random_phase(self):
        """Test random phase generation with key."""
        key = jax.random.PRNGKey(42)
        grating1 = make_drifting_grating(key=key, image_duration=10)

        key2 = jax.random.PRNGKey(123)
        grating2 = make_drifting_grating(key=key2, image_duration=10)

        # Different keys should give different gratings
        assert not jnp.allclose(grating1, grating2)

    def test_grating_fixed_phase(self):
        """Test fixed phase reproducibility."""
        grating1 = make_drifting_grating(phase=45.0, image_duration=10)
        grating2 = make_drifting_grating(phase=45.0, image_duration=10)

        assert jnp.allclose(grating1, grating2)

    def test_grating_invalid_contrast(self):
        """Test error on invalid contrast."""
        with pytest.raises(ValueError):
            make_drifting_grating(contrast=1.5)

        with pytest.raises(ValueError):
            make_drifting_grating(contrast=0.0)


class TestGratingWithDelays:
    """Tests for grating stimulus with pre/post delays."""

    def test_grating_with_delays_shape(self):
        """Test output shape includes delays."""
        config = StimulusConfig(
            pre_delay=50,
            im_slice=100,
            post_delay=150,
        )
        video = generate_grating_with_delays(
            orientation=45.0,
            config=config,
        )
        # Total length = pre + im + post = 300
        expected_len = config.pre_delay + config.im_slice + config.post_delay
        assert video.shape[0] == expected_len

    def test_grating_with_delays_structure(self):
        """Test stimulus has zeros in delay periods."""
        config = StimulusConfig(
            pre_delay=50,
            im_slice=100,
            post_delay=50,
            intensity=2.0,
        )
        video = generate_grating_with_delays(orientation=45.0, config=config)

        # Pre-delay should be zeros
        pre_region = video[:config.pre_delay]
        assert jnp.allclose(pre_region, 0.0)

        # Post-delay should be zeros
        post_region = video[-config.post_delay:]
        assert jnp.allclose(post_region, 0.0)

        # Image region should have non-zero values
        im_region = video[config.pre_delay:config.pre_delay + config.im_slice]
        assert jnp.abs(im_region).max() > 0


class TestFiringRateConversion:
    """Tests for firing rate to input conversion."""

    def test_current_input_scaling(self):
        """Test current input mode scales appropriately."""
        rates = jnp.array([[100.0, 200.0], [50.0, 150.0]])

        inputs = firing_rates_to_input(rates, current_input=True, scale=1.3)

        # Should be p * scale where p = 1 - exp(-rate/1000)
        expected_p = 1.0 - jnp.exp(-rates / 1000.0)
        expected = expected_p * 1.3
        assert jnp.allclose(inputs, expected, rtol=1e-5)

    def test_spike_input_binary(self):
        """Test spike input mode produces binary values."""
        rates = jnp.full((100, 50), 500.0)
        key = jax.random.PRNGKey(42)

        inputs = firing_rates_to_input(rates, current_input=False, key=key)

        # Should be binary
        assert jnp.all((inputs == 0.0) | (inputs == 1.0))

    def test_spike_rate_correlation(self):
        """Test higher rates produce more spikes on average."""
        low_rates = jnp.full((1000, 50), 10.0)
        high_rates = jnp.full((1000, 50), 500.0)
        key = jax.random.PRNGKey(42)

        low_spikes = firing_rates_to_input(low_rates, current_input=False, key=key)
        key2 = jax.random.PRNGKey(43)
        high_spikes = firing_rates_to_input(high_rates, current_input=False, key=key2)

        assert jnp.mean(low_spikes) < jnp.mean(high_spikes)


class TestLabelGeneration:
    """Tests for classification label generation."""

    def test_classification_labels_shape(self):
        """Test label shapes are correct."""
        labels, weights = create_classification_labels(
            class_label=5,
            pre_chunks=2,
            resp_chunks=1,
            post_chunks=3,
        )

        assert labels.shape == (6,)
        assert weights.shape == (6,)

    def test_classification_labels_values(self):
        """Test label values are correct."""
        labels, weights = create_classification_labels(
            class_label=3,
            pre_chunks=2,
            resp_chunks=2,
            post_chunks=1,
        )

        # Pre-chunks should be 0
        assert jnp.all(labels[:2] == 0)

        # Response chunks should be class_label
        assert jnp.all(labels[2:4] == 3)

        # Post-chunks should be 0
        assert jnp.all(labels[4:] == 0)

        # Weights should be 1 only in response window
        assert jnp.all(weights[:2] == 0)
        assert jnp.all(weights[2:4] == 1)
        assert jnp.all(weights[4:] == 0)

    def test_image_labels(self):
        """Test image labels for visualization."""
        img_labels = create_image_labels(
            image_index=7,
            pre_delay=50,
            im_slice=100,
            post_delay=50,
            chunk_size=50,
        )

        # Shape: (pre + im + post) / chunk_size = (50 + 100 + 50) / 50 = 4
        assert img_labels.shape == (4,)

        # Pre: 0, Im: 7, 7, Post: 0
        expected = jnp.array([0, 7, 7, 0])
        assert jnp.all(img_labels == expected)


class TestImagePreparation:
    """Tests for image preprocessing."""

    def test_prepare_grayscale_image(self):
        """Test grayscale image preparation."""
        image = jnp.ones((28, 28)) * 128  # Simple grayscale

        prepared = prepare_image_for_lgn(image, intensity=2.0)

        assert prepared.shape == (120, 240, 1)

    def test_prepare_rgb_image(self):
        """Test RGB image preparation."""
        image = jnp.ones((32, 32, 3)) * 128

        prepared = prepare_image_for_lgn(image, intensity=2.0)

        assert prepared.shape == (120, 240, 1)

    def test_prepare_image_normalization(self):
        """Test image normalization to intensity range."""
        # Black image
        black = jnp.zeros((28, 28))
        prepared_black = prepare_image_for_lgn(black, intensity=2.0)
        # Should map to -intensity
        assert jnp.isclose(prepared_black.mean(), -2.0, atol=0.1)

        # White image
        white = jnp.ones((28, 28)) * 255
        prepared_white = prepare_image_for_lgn(white, intensity=2.0)
        # Should map to +intensity
        assert jnp.isclose(prepared_white.mean(), 2.0, atol=0.1)


class TestStimulusSequence:
    """Tests for stimulus sequence creation."""

    def test_sequence_shape(self):
        """Test stimulus sequence shape."""
        image = jnp.ones((120, 240, 1))

        sequence = create_stimulus_sequence(
            image,
            pre_delay=50,
            im_slice=100,
            post_delay=50,
        )

        assert sequence.shape == (200, 120, 240, 1)

    def test_sequence_structure(self):
        """Test stimulus sequence has correct structure."""
        image = jnp.ones((120, 240, 1))

        sequence = create_stimulus_sequence(
            image,
            pre_delay=50,
            im_slice=100,
            post_delay=50,
        )

        # Pre-delay zeros
        assert jnp.allclose(sequence[:50], 0.0)

        # Image region
        assert jnp.allclose(sequence[50:150], 1.0)

        # Post-delay zeros
        assert jnp.allclose(sequence[150:], 0.0)


class TestChunkComputation:
    """Tests for chunk computation utilities."""

    def test_compute_n_chunks(self):
        """Test chunk computation."""
        total, pre, im, post = compute_n_chunks(
            im_slice=100,
            pre_delay=50,
            post_delay=150,
            chunk_size=50,
        )

        assert total == 6
        assert pre == 1
        assert im == 2
        assert post == 3

    def test_validate_config_valid(self):
        """Test validation passes for valid config."""
        config = StimulusConfig(
            pre_delay=50,
            im_slice=100,
            post_delay=150,
            chunk_size=50,
        )
        validate_config(config)  # Should not raise

    def test_validate_config_invalid_divisibility(self):
        """Test validation fails for indivisible sequence."""
        config = StimulusConfig(
            pre_delay=55,  # Not divisible by 50
            im_slice=100,
            post_delay=150,
            chunk_size=50,
        )
        with pytest.raises(ValueError):
            validate_config(config)

    def test_validate_config_invalid_intensity(self):
        """Test validation fails for invalid intensity."""
        config = StimulusConfig(intensity=-1.0)
        with pytest.raises(ValueError):
            validate_config(config)


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_grating_precompute(self):
        """Test grating generation works for precomputation.

        Note: make_drifting_grating uses numpy internally and is not
        designed for JIT compilation. It's meant for pre-generating
        stimuli before training.
        """
        key = jax.random.PRNGKey(42)
        grating = make_drifting_grating(
            image_duration=10,
            key=key,
        )
        assert grating.ndim == 4
        assert grating.shape[0] == 10

    def test_firing_rates_jit(self):
        """Test firing rate conversion is JIT compatible."""
        @jax.jit
        def convert(rates, key):
            return firing_rates_to_input(rates, current_input=False, key=key)

        rates = jnp.ones((100, 50)) * 100
        key = jax.random.PRNGKey(42)
        inputs = convert(rates, key)
        assert inputs.shape == rates.shape
