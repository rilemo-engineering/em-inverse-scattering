"""
Unit tests for data/mat_io.py - MATLAB .mat file I/O.

Tests cover:
- Basic load/save operations
- Round-trip integrity
- Scenario data loading
- Experimental data loading
- Error handling
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from inverse_scattering.data.mat_io import (
    load_mat,
    save_mat,
    load_scenario_data,
    load_experimental_scenario,
    load_object_data,
    save_scenario_data,
)


class TestLoadMat:
    """Tests for load_mat function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_mat("/nonexistent/path/file.mat")

    def test_basic_load(self, tmp_path):
        """Should load basic .mat file."""
        from scipy.io import savemat

        # Create test file
        filepath = tmp_path / "test.mat"
        data = {'x': np.array([1, 2, 3]), 'y': np.array([[1, 2], [3, 4]])}
        savemat(str(filepath), data)

        # Load
        loaded = load_mat(filepath)

        assert 'x' in loaded
        assert 'y' in loaded

    def test_removes_metadata_keys(self, tmp_path):
        """Should remove MATLAB metadata keys (starting with __)."""
        from scipy.io import savemat

        filepath = tmp_path / "test.mat"
        savemat(str(filepath), {'var': np.array([1, 2, 3])})

        loaded = load_mat(filepath)

        # Should not have __header__, __version__, __globals__
        for key in loaded.keys():
            assert not key.startswith('__')

    def test_squeeze_me_true(self, tmp_path):
        """With squeeze_me=True, singleton dimensions should be removed."""
        from scipy.io import savemat

        filepath = tmp_path / "test.mat"
        # Save (1,5) array
        savemat(str(filepath), {'vec': np.array([[1, 2, 3, 4, 5]])})

        loaded = load_mat(filepath, squeeze_me=True)

        # Should be squeezed to (5,)
        assert loaded['vec'].ndim == 1
        assert loaded['vec'].shape == (5,)

    def test_squeeze_me_false(self, tmp_path):
        """With squeeze_me=False, dimensions should be preserved."""
        from scipy.io import savemat

        filepath = tmp_path / "test.mat"
        savemat(str(filepath), {'vec': np.array([[1, 2, 3, 4, 5]])})

        loaded = load_mat(filepath, squeeze_me=False)

        # Should keep (1, 5) shape
        assert loaded['vec'].ndim == 2

    def test_complex_data(self, tmp_path):
        """Should correctly load complex data."""
        from scipy.io import savemat

        filepath = tmp_path / "test.mat"
        z = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        savemat(str(filepath), {'z': z})

        loaded = load_mat(filepath)

        np.testing.assert_array_almost_equal(loaded['z'], z)

    def test_accepts_path_object(self, tmp_path):
        """Should accept Path object as filepath."""
        from scipy.io import savemat

        filepath = tmp_path / "test.mat"
        savemat(str(filepath), {'x': np.array([1, 2, 3])})

        # Pass Path object
        loaded = load_mat(Path(filepath))

        assert 'x' in loaded


class TestSaveMat:
    """Tests for save_mat function."""

    def test_basic_save(self, tmp_path):
        """Should save data to .mat file."""
        filepath = tmp_path / "test.mat"
        data = {'x': np.array([1, 2, 3])}

        save_mat(filepath, data)

        assert filepath.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if needed."""
        filepath = tmp_path / "subdir" / "nested" / "test.mat"
        data = {'x': np.array([1, 2, 3])}

        save_mat(filepath, data)

        assert filepath.exists()

    def test_round_trip_real(self, tmp_path):
        """Save then load should preserve real data."""
        filepath = tmp_path / "test.mat"
        original = {'x': np.array([1.0, 2.0, 3.0]),
                    'y': np.array([[1, 2], [3, 4]])}

        save_mat(filepath, original)
        loaded = load_mat(filepath)

        np.testing.assert_array_equal(loaded['x'], original['x'])
        np.testing.assert_array_equal(loaded['y'], original['y'])

    def test_round_trip_complex(self, tmp_path):
        """Save then load should preserve complex data."""
        filepath = tmp_path / "test.mat"
        original = {'z': np.array([1 + 2j, 3 + 4j, 5 + 6j])}

        save_mat(filepath, original)
        loaded = load_mat(filepath)

        np.testing.assert_array_almost_equal(loaded['z'], original['z'])

    def test_oned_as_column(self, tmp_path):
        """oned_as='column' should save 1D arrays as column vectors."""
        from scipy.io import loadmat

        filepath = tmp_path / "test.mat"
        data = {'vec': np.array([1, 2, 3])}

        save_mat(filepath, data, oned_as='column')

        # Load without squeeze to check shape
        loaded = loadmat(str(filepath), squeeze_me=False)

        # Should be (3, 1) column vector
        assert loaded['vec'].shape == (3, 1)

    def test_oned_as_row(self, tmp_path):
        """oned_as='row' should save 1D arrays as row vectors."""
        from scipy.io import loadmat

        filepath = tmp_path / "test.mat"
        data = {'vec': np.array([1, 2, 3])}

        save_mat(filepath, data, oned_as='row')

        # Load without squeeze to check shape
        loaded = loadmat(str(filepath), squeeze_me=False)

        # Should be (1, 3) row vector
        assert loaded['vec'].shape == (1, 3)


class TestLoadScenarioData:
    """Tests for load_scenario_data function."""

    def test_loads_expected_keys(self, tmp_path):
        """Should load and return expected scenario keys."""
        from scipy.io import savemat

        filepath = tmp_path / "scenario.mat"

        # Create scenario data
        Nx, Ny = 10, 10
        Nm, Nv = 5, 5
        data = {
            'Escat': np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv),
            'PROF': np.random.randn(Ny, Nx),
            'Einc_domain': np.random.randn(Ny, Nx, Nv),
            'Etot_domain': np.random.randn(Ny, Nx, Nv),
            'freq': 1e9,
            'lx': 0.1,
            'ly': 0.1,
            'Nx': Nx,
            'Ny': Ny,
            'eb': 1.0,
            'sb': 0.0,
            'Rm': 0.5,
            'DOF': 25,
        }
        savemat(str(filepath), data)

        loaded = load_scenario_data(filepath)

        assert 'Escat' in loaded
        assert 'PROF' in loaded
        assert 'freq' in loaded

    def test_preserves_array_shapes(self, tmp_path):
        """Should preserve array shapes correctly."""
        from scipy.io import savemat

        filepath = tmp_path / "scenario.mat"

        Nx, Ny = 8, 8
        Nm, Nv = 4, 4
        data = {
            'Escat': np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv),
            'PROF': np.random.randn(Ny, Nx),
            'freq': 1e9,
            'Nx': Nx,
            'Ny': Ny,
        }
        savemat(str(filepath), data)

        loaded = load_scenario_data(filepath)

        assert loaded['Escat'].shape == (Nm, Nv)
        assert loaded['PROF'].shape == (Ny, Nx)


class TestLoadExperimentalScenario:
    """Tests for load_experimental_scenario function."""

    def test_delegates_to_load_scenario(self, tmp_path):
        """Should use load_scenario_data internally."""
        from scipy.io import savemat

        filepath = tmp_path / "scenario_exp.mat"
        data = {
            'Escat': np.array([[1, 2], [3, 4]]),
            'freq': 1e9,
        }
        savemat(str(filepath), data)

        loaded = load_experimental_scenario(filepath)

        assert 'Escat' in loaded


class TestLoadObjectData:
    """Tests for load_object_data function."""

    def test_loads_object_file(self, tmp_path):
        """Should load object specification file."""
        from scipy.io import savemat

        filepath = tmp_path / "object.mat"
        data = {
            'r0': 0.05,
            'x0': 0.0,
            'y0': 0.0,
            'PROF': np.random.randn(10, 10),
        }
        savemat(str(filepath), data)

        loaded = load_object_data(filepath)

        assert 'r0' in loaded
        assert 'x0' in loaded
        assert 'PROF' in loaded


class TestSaveScenarioData:
    """Tests for save_scenario_data function."""

    def test_saves_all_required_fields(self, tmp_path):
        """Should save all required scenario fields."""
        filepath = tmp_path / "scenario.mat"

        Nx, Ny = 8, 8
        Nm, Nv = 4, 4
        Escat = np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv)
        PROF = np.random.randn(Ny, Nx)
        Einc_domain = np.random.randn(Ny, Nx, Nv)
        Etot_domain = np.random.randn(Ny, Nx, Nv)

        save_scenario_data(
            filepath=filepath,
            Escat=Escat,
            PROF=PROF,
            Einc_domain=Einc_domain,
            Etot_domain=Etot_domain,
            freq=1e9,
            lx=0.1,
            ly=0.1,
            Nx=Nx,
            Ny=Ny,
            eb=1.0,
            sb=0.0,
            Rm=0.5,
            DOF=25,
        )

        loaded = load_scenario_data(filepath)

        assert 'Escat' in loaded
        assert 'PROF' in loaded
        assert 'freq' in loaded
        assert 'Nx' in loaded

    def test_saves_additional_kwargs(self, tmp_path):
        """Should save additional keyword arguments."""
        filepath = tmp_path / "scenario.mat"

        Nx, Ny = 4, 4
        save_scenario_data(
            filepath=filepath,
            Escat=np.zeros((2, 2)),
            PROF=np.zeros((Ny, Nx)),
            Einc_domain=np.zeros((Ny, Nx, 2)),
            Etot_domain=np.zeros((Ny, Nx, 2)),
            freq=1e9,
            lx=0.1,
            ly=0.1,
            Nx=Nx,
            Ny=Ny,
            eb=1.0,
            sb=0.0,
            Rm=0.5,
            DOF=16,
            custom_var=np.array([1, 2, 3]),
        )

        loaded = load_mat(filepath)

        assert 'custom_var' in loaded

    def test_round_trip_scenario(self, tmp_path):
        """Save then load scenario should preserve data."""
        filepath = tmp_path / "scenario.mat"

        Nx, Ny = 6, 6
        Nm, Nv = 3, 3
        Escat = np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv)
        PROF = np.random.randn(Ny, Nx) + 0.1j * np.random.randn(Ny, Nx)

        save_scenario_data(
            filepath=filepath,
            Escat=Escat,
            PROF=PROF,
            Einc_domain=np.zeros((Ny, Nx, Nv)),
            Etot_domain=np.zeros((Ny, Nx, Nv)),
            freq=1e9,
            lx=0.1,
            ly=0.1,
            Nx=Nx,
            Ny=Ny,
            eb=1.0,
            sb=0.0,
            Rm=0.5,
            DOF=36,
        )

        loaded = load_scenario_data(filepath)

        np.testing.assert_array_almost_equal(loaded['Escat'], Escat)
        np.testing.assert_array_almost_equal(loaded['PROF'], PROF)


class TestRealMATLABFiles:
    """Tests using real MATLAB files from the exercise data."""

    @pytest.fixture
    def simulated_data_path(self):
        """Path to simulated data directory."""
        return Path(__file__).parent.parent.parent.parent / "matlab" / "simulated"

    @pytest.fixture
    def experimental_data_path(self):
        """Path to experimental data directory."""
        return Path(__file__).parent.parent.parent.parent / "matlab" / "experimental"

    @pytest.fixture
    def scenario_mat_path(self, simulated_data_path):
        """Path to DATA_scenario.mat in scenario folder."""
        return simulated_data_path / "scenario" / "DATA_scenario.mat"

    @pytest.fixture
    def inversion_mat_path(self, simulated_data_path):
        """Path to inversion folder."""
        return simulated_data_path / "inversion"

    @pytest.fixture
    def exp_inversion_mat_path(self, experimental_data_path):
        """Path to experimental inversion folder."""
        return experimental_data_path / "inversion"

    def test_load_data_scenario(self, scenario_mat_path):
        """Test loading DATA_scenario.mat from scenario folder."""
        if not scenario_mat_path.exists():
            pytest.skip("MATLAB reference files not available")

        data = load_scenario_data(scenario_mat_path)

        # Check expected variables exist
        assert 'Escat' in data
        assert 'PROF' in data or 'prof' in {k.lower() for k in data.keys()}

        # scenario Escat may be placeholder (zeros) before computation
        # Just check it exists with expected shape
        assert data['Escat'].ndim == 2

    def test_load_data_scenario_inversion(self, inversion_mat_path):
        """Test loading DATA_scenario.mat from inversion folder."""
        filepath = inversion_mat_path / "DATA_scenario.mat"
        if not filepath.exists():
            pytest.skip("MATLAB reference files not available")

        data = load_scenario_data(filepath)

        assert 'Escat' in data
        assert data['Escat'].ndim == 2  # (Nm, Nv)

    def test_load_data_scenario_noweak(self, inversion_mat_path):
        """Test loading DATA_scenario_noweak.mat (strong scatterer)."""
        filepath = inversion_mat_path / "DATA_scenario_noweak.mat"
        if not filepath.exists():
            pytest.skip("MATLAB reference files not available")

        data = load_scenario_data(filepath)

        assert 'Escat' in data

    def test_load_data_scenario_square(self, inversion_mat_path):
        """Test loading DATA_scenario_square.mat (square target)."""
        filepath = inversion_mat_path / "DATA_scenario_square.mat"
        if not filepath.exists():
            pytest.skip("MATLAB reference files not available")

        data = load_scenario_data(filepath)

        assert 'Escat' in data

    def test_load_experimental_twotargets(self, exp_inversion_mat_path):
        """Test loading experimental two-targets scenario."""
        filepath = exp_inversion_mat_path / "DATA_scenario_exp_twotargets.mat"
        if not filepath.exists():
            pytest.skip("Experimental data files not available")

        data = load_experimental_scenario(filepath)

        assert 'Escat' in data

    def test_load_object_twotargets(self, exp_inversion_mat_path):
        """Test loading experimental two-targets object specification."""
        filepath = exp_inversion_mat_path / "DATA_object_exp_twotargets.mat"
        if not filepath.exists():
            pytest.skip("Experimental data files not available")

        data = load_object_data(filepath)

        # Should have target geometry info
        assert len(data) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dict(self, tmp_path):
        """Should handle empty data dictionary."""
        filepath = tmp_path / "empty.mat"

        save_mat(filepath, {})
        loaded = load_mat(filepath)

        assert isinstance(loaded, dict)

    def test_scalar_values(self, tmp_path):
        """Should handle scalar values."""
        filepath = tmp_path / "scalars.mat"

        data = {
            'scalar_int': 42,
            'scalar_float': 3.14159,
            'scalar_complex': 1 + 2j,
        }
        save_mat(filepath, data)
        loaded = load_mat(filepath)

        assert loaded['scalar_int'] == 42
        assert abs(loaded['scalar_float'] - 3.14159) < 1e-5

    def test_large_array(self, tmp_path):
        """Should handle larger arrays."""
        filepath = tmp_path / "large.mat"

        large_array = np.random.randn(100, 100)
        save_mat(filepath, {'large': large_array})
        loaded = load_mat(filepath)

        np.testing.assert_array_almost_equal(loaded['large'], large_array)

    def test_special_characters_in_path(self, tmp_path):
        """Should handle paths with spaces (via tmp_path which escapes properly)."""
        # tmp_path handles this appropriately
        filepath = tmp_path / "test file.mat"

        save_mat(filepath, {'x': np.array([1, 2, 3])})
        loaded = load_mat(filepath)

        assert 'x' in loaded
