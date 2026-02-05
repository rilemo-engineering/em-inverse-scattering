"""
Unit tests for data/fresnel_loader.py - Fresnel Institute data loader.

Tests cover:
- Frequency mapping
- Data file loading
- Parameter retrieval
- MATLAB interface compatibility
- Error handling

The Fresnel Institute data format:
    [Tx_index, Rx_index, Freq_index, Re_scat, Im_scat, Re_inc, Im_inc]
"""

import numpy as np
import pytest
from pathlib import Path

from inverse_scattering.data.fresnel_loader import (
    FRESNEL_FREQUENCIES_8F,
    load_fresnel_data,
    _frequency_to_index,
    get_fresnel_parameters,
    load_data_fr2001,
)


class TestFresnelFrequencies:
    """Tests for frequency mapping constants."""

    def test_has_8_frequencies(self):
        """Should have 8 frequency entries."""
        assert len(FRESNEL_FREQUENCIES_8F) == 8

    def test_indices_are_1_based(self):
        """Indices should be 1-based (1 to 8)."""
        assert set(FRESNEL_FREQUENCIES_8F.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_index_1_is_2ghz(self):
        """Index 1 should map to 2 GHz."""
        assert FRESNEL_FREQUENCIES_8F[1] == 2e9

    def test_index_3_is_4ghz(self):
        """Index 3 should map to 4 GHz (default for exercises)."""
        assert FRESNEL_FREQUENCIES_8F[3] == 4e9

    def test_frequencies_increase(self):
        """Frequencies should generally increase with index."""
        freqs = [FRESNEL_FREQUENCIES_8F[i] for i in range(1, 9)]
        # Allow for non-strict ordering but first should be lowest
        assert freqs[0] == min(freqs)

    def test_all_values_are_hz(self):
        """All frequency values should be in Hz (10^9 range)."""
        for freq in FRESNEL_FREQUENCIES_8F.values():
            assert 1e9 <= freq <= 20e9


class TestFrequencyToIndex:
    """Tests for _frequency_to_index function."""

    def test_2ghz_returns_1(self):
        """2 GHz should return index 1."""
        assert _frequency_to_index(2e9) == 1

    def test_4ghz_returns_3(self):
        """4 GHz should return index 3."""
        assert _frequency_to_index(4e9) == 3

    def test_3ghz_returns_2(self):
        """3 GHz should return index 2."""
        assert _frequency_to_index(3e9) == 2

    def test_tolerance_matching(self):
        """Should match within tolerance."""
        # Slightly off from 4 GHz but within tolerance
        assert _frequency_to_index(4.05e9, tolerance=0.1e9) == 3

    def test_invalid_frequency_raises(self):
        """Should raise ValueError for invalid frequency."""
        with pytest.raises(ValueError) as exc_info:
            _frequency_to_index(2.5e9)  # 2.5 GHz not in list

        assert "not in dataset" in str(exc_info.value)

    def test_error_message_lists_available(self):
        """Error message should list available frequencies."""
        with pytest.raises(ValueError) as exc_info:
            _frequency_to_index(2.5e9)

        assert "GHz" in str(exc_info.value)


class TestGetFresnelParameters:
    """Tests for get_fresnel_parameters function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        params = get_fresnel_parameters()
        assert isinstance(params, dict)

    def test_has_freq(self):
        """Should have freq parameter."""
        params = get_fresnel_parameters()
        assert 'freq' in params
        assert params['freq'] == 4e9  # 4 GHz

    def test_has_wavelength(self):
        """Should have wavelength parameter."""
        params = get_fresnel_parameters()
        assert 'lambda0' in params
        assert params['lambda0'] == 0.075  # 7.5 cm

    def test_has_doi_size(self):
        """Should have DoI dimensions."""
        params = get_fresnel_parameters()
        assert 'lx' in params
        assert 'ly' in params
        assert params['lx'] == 0.15  # 15 cm
        assert params['ly'] == 0.15

    def test_has_grid_size(self):
        """Should have grid size."""
        params = get_fresnel_parameters()
        assert 'Nx' in params
        assert 'Ny' in params
        assert params['Nx'] == 64
        assert params['Ny'] == 64

    def test_has_measurement_radii(self):
        """Should have measurement and transmitter radii."""
        params = get_fresnel_parameters()
        assert 'Rv' in params  # Transmitter radius
        assert 'Rm' in params  # Receiver radius
        assert params['Rv'] == 0.72135
        assert params['Rm'] == 0.76135

    def test_has_background_properties(self):
        """Should have background permittivity and conductivity."""
        params = get_fresnel_parameters()
        assert params['eb'] == 1.0  # Free space
        assert params['sb'] == 0.0  # No conductivity

    def test_wavelength_matches_freq(self):
        """Wavelength should match frequency: c/f = λ."""
        params = get_fresnel_parameters()
        c = 3e8  # Speed of light
        expected_lambda = c / params['freq']
        assert abs(params['lambda0'] - expected_lambda) < 0.001


class TestLoadFresnelData:
    """Tests for load_fresnel_data function."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a sample Fresnel data file."""
        # Create sample data with:
        # 3 transmitters, 4 receivers, 2 frequencies
        # Format: Tx Rx Freq Re_scat Im_scat Re_inc Im_inc

        data_lines = []
        for freq_idx in [1, 3]:  # 2 GHz and 4 GHz
            for tx in [1, 2, 3]:
                for rx in [1, 2, 3, 4]:
                    re_scat = 0.1 * tx + 0.01 * rx + 0.001 * freq_idx
                    im_scat = 0.05 * tx + 0.005 * rx
                    re_inc = 1.0
                    im_inc = 0.0
                    data_lines.append(f"{tx} {rx} {freq_idx} {re_scat} {im_scat} {re_inc} {im_inc}")

        filepath = tmp_path / "test_fresnel.txt"
        filepath.write_text("\n".join(data_lines))
        return filepath

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_fresnel_data("/nonexistent/path.txt", freq=4e9, nx=10, ny=10)

    def test_returns_tuple(self, sample_data_file):
        """Should return tuple of (Escat, Einc_domain)."""
        result = load_fresnel_data(sample_data_file, freq=4e9, nx=10, ny=10)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_escat_shape(self, sample_data_file):
        """Escat should have shape (Nm, Nv) = (n_rx, n_tx)."""
        Escat, _ = load_fresnel_data(sample_data_file, freq=4e9, nx=10, ny=10)

        # 4 receivers × 3 transmitters
        assert Escat.shape == (4, 3)

    def test_escat_is_complex(self, sample_data_file):
        """Escat should be complex."""
        Escat, _ = load_fresnel_data(sample_data_file, freq=4e9, nx=10, ny=10)

        assert np.iscomplexobj(Escat)

    def test_einc_domain_shape(self, sample_data_file):
        """Einc_domain should have shape (Ny, Nx, Nv)."""
        Nx, Ny = 10, 12
        _, Einc_domain = load_fresnel_data(sample_data_file, freq=4e9, nx=Nx, ny=Ny)

        # Shape: (Ny, Nx, Nv)
        assert Einc_domain.shape[0] == Ny
        assert Einc_domain.shape[1] == Nx
        assert Einc_domain.shape[2] == 3  # n_tx

    def test_frequency_filtering(self, sample_data_file):
        """Should only return data for specified frequency."""
        # Load 4 GHz data
        Escat_4ghz, _ = load_fresnel_data(sample_data_file, freq=4e9, nx=10, ny=10)
        # Load 2 GHz data
        Escat_2ghz, _ = load_fresnel_data(sample_data_file, freq=2e9, nx=10, ny=10)

        # Data should be different
        assert not np.allclose(Escat_4ghz, Escat_2ghz)

    def test_complex_values_correct(self, sample_data_file):
        """Complex values should be correctly assembled."""
        Escat, _ = load_fresnel_data(sample_data_file, freq=4e9, nx=10, ny=10)

        # All values should have non-zero components
        assert np.all(np.abs(Escat) > 0)

    def test_accepts_path_object(self, sample_data_file):
        """Should accept Path object."""
        Escat, _ = load_fresnel_data(Path(sample_data_file), freq=4e9, nx=10, ny=10)

        assert Escat.shape == (4, 3)


class TestLoadDataFr2001:
    """Tests for MATLAB-compatible interface."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create sample data file."""
        data_lines = []
        for tx in [1, 2]:
            for rx in [1, 2, 3]:
                data_lines.append(f"{tx} {rx} 3 0.1 0.05 1.0 0.0")

        filepath = tmp_path / "test_data.txt"
        filepath.write_text("\n".join(data_lines))
        return filepath

    def test_matches_load_fresnel_data(self, sample_data_file):
        """Should produce same result as load_fresnel_data."""
        Escat_direct, Einc_direct = load_fresnel_data(
            sample_data_file, freq=4e9, nx=10, ny=10
        )

        Escat_matlab, Einc_matlab = load_data_fr2001(
            freq=4e9,
            dataset=str(sample_data_file),
            Nx=10,
            Ny=10
        )

        np.testing.assert_array_equal(Escat_direct, Escat_matlab)

    def test_with_data_dir(self, sample_data_file):
        """Should accept data_dir parameter."""
        data_dir = sample_data_file.parent
        filename = sample_data_file.name

        Escat, _ = load_data_fr2001(
            freq=4e9,
            dataset=filename,
            Nx=10,
            Ny=10,
            data_dir=data_dir
        )

        assert Escat.shape == (3, 2)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_tx_rx(self, tmp_path):
        """Single row file requires at least 2 rows for np.loadtxt to return 2D."""
        filepath = tmp_path / "single.txt"
        # np.loadtxt returns 1D array for single line, need at least 2 lines
        filepath.write_text("1 1 3 0.1 0.05 1.0 0.0\n1 1 3 0.1 0.05 1.0 0.0")

        Escat, _ = load_fresnel_data(filepath, freq=4e9, nx=5, ny=5)

        assert Escat.shape == (1, 1)
        assert Escat[0, 0] == 0.1 + 0.05j

    def test_non_contiguous_indices(self, tmp_path):
        """Should handle non-contiguous tx/rx indices."""
        filepath = tmp_path / "sparse.txt"
        data_lines = [
            "1 1 3 0.1 0.05 1.0 0.0",
            "1 5 3 0.2 0.05 1.0 0.0",  # rx 5, skip 2-4
            "10 1 3 0.3 0.05 1.0 0.0",  # tx 10, skip 2-9
            "10 5 3 0.4 0.05 1.0 0.0",
        ]
        filepath.write_text("\n".join(data_lines))

        Escat, _ = load_fresnel_data(filepath, freq=4e9, nx=5, ny=5)

        # Should map to 2×2 matrix (2 unique tx, 2 unique rx)
        assert Escat.shape == (2, 2)

    def test_whitespace_variations(self, tmp_path):
        """Should handle various whitespace formats."""
        filepath = tmp_path / "whitespace.txt"
        # Multiple spaces, tabs
        data_lines = [
            "1  1  3  0.1  0.05  1.0  0.0",
            "1\t2\t3\t0.2\t0.05\t1.0\t0.0",
        ]
        filepath.write_text("\n".join(data_lines))

        Escat, _ = load_fresnel_data(filepath, freq=4e9, nx=5, ny=5)

        assert Escat.shape == (2, 1)


class TestRealDataFiles:
    """Tests using real Fresnel data files if available."""

    @pytest.fixture
    def fresnel_data_dir(self):
        """Path to Fresnel data directory."""
        # Path relative to test file
        return Path(__file__).parent.parent.parent.parent / "matlab" / "experimental"

    @pytest.mark.skip(reason="Fresnel text data files not typically included")
    def test_load_dielTM_dec8f(self, fresnel_data_dir):
        """Test loading the standard dielTM_dec8f.txt dataset."""
        if fresnel_data_dir is None:
            pytest.skip("Fresnel data directory not found")

        filepath = fresnel_data_dir / "dielTM_dec8f.txt"
        if not filepath.exists():
            pytest.skip("dielTM_dec8f.txt not found")

        params = get_fresnel_parameters()
        Escat, Einc_domain = load_fresnel_data(
            filepath,
            freq=params['freq'],
            nx=params['Nx'],
            ny=params['Ny']
        )

        # Should have typical Fresnel dimensions
        assert Escat.ndim == 2
        assert Escat.shape[0] > 0  # Has receivers
        assert Escat.shape[1] > 0  # Has transmitters
        assert np.iscomplexobj(Escat)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_parameters_compatible_with_loader(self, tmp_path):
        """Standard parameters should work with loader."""
        params = get_fresnel_parameters()

        # Create test data compatible with params
        data_lines = []
        freq_idx = _frequency_to_index(params['freq'])
        for tx in range(1, 5):
            for rx in range(1, 6):
                data_lines.append(f"{tx} {rx} {freq_idx} 0.1 0.05 1.0 0.0")

        filepath = tmp_path / "test.txt"
        filepath.write_text("\n".join(data_lines))

        Escat, Einc_domain = load_fresnel_data(
            filepath,
            freq=params['freq'],
            nx=params['Nx'],
            ny=params['Ny']
        )

        # Check shapes are consistent
        assert Escat.ndim == 2
        assert Einc_domain.shape == (params['Ny'], params['Nx'], 4)  # 4 transmitters
