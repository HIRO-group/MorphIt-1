"""
Test script for MorphIt system.
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from config import get_config, update_config_from_dict

        print("✓ config module imported successfully")
    except Exception as e:
        print(f"✗ config module import failed: {e}")
        return False

    try:
        from morphit import MorphIt

        print("✓ morphit module imported successfully")
    except Exception as e:
        print(f"✗ morphit module import failed: {e}")
        return False

    try:
        from losses import MorphItLosses

        print("✓ losses module imported successfully")
    except Exception as e:
        print(f"✗ losses module import failed: {e}")
        return False

    try:
        from density_control import DensityController

        print("✓ density_control module imported successfully")
    except Exception as e:
        print(f"✗ density_control module import failed: {e}")
        return False

    try:
        from training import MorphItTrainer, train_morphit

        print("✓ training module imported successfully")
    except Exception as e:
        print(f"✗ training module import failed: {e}")
        return False

    try:
        from convergence_tracker import ConvergenceTracker

        print("✓ convergence_tracker module imported successfully")
    except Exception as e:
        print(f"✗ convergence_tracker module import failed: {e}")
        return False

    try:
        from logger import SphereEvolutionLogger

        print("✓ logger module imported successfully")
    except Exception as e:
        print(f"✗ logger module import failed: {e}")
        return False

    try:
        from visualization import visualize_packing

        print("✓ visualization module imported successfully")
    except Exception as e:
        print(f"✗ visualization module import failed: {e}")
        return False

    return True


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")

    try:
        from config import get_config, update_config_from_dict

        # Test default config
        config = get_config()
        print("✓ Default config created successfully")

        # Test different configurations
        for config_name in ["MorphIt-V", "MorphIt-S", "MorphIt-B"]:
            config = get_config(config_name)
            print(f"✓ {config_name} config created successfully")

        # Test config updates
        config = get_config("MorphIt-B")
        updates = {
            "model.num_spheres": 10,
            "training.iterations": 25,
            "training.verbose_frequency": 5,
        }
        config = update_config_from_dict(config, updates)

        assert config.model.num_spheres == 10
        assert config.training.iterations == 25
        assert config.training.verbose_frequency == 5
        print("✓ Config updates work correctly")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_model_creation():
    """Test basic model creation."""
    print("\nTesting model creation...")

    try:
        from morphit import MorphIt
        from config import get_config

        # Test with default config
        model = MorphIt()
        print("✓ Model created with default config")

        # Test with custom config
        config = get_config("MorphIt-B")
        config.model.num_spheres = 5  # Small number for testing
        model = MorphIt(config)
        print("✓ Model created with custom config")

        # Test model properties
        assert hasattr(model, "centers")
        assert hasattr(model, "radii")
        assert hasattr(model, "query_mesh")
        assert hasattr(model, "inside_samples")
        assert hasattr(model, "surface_samples")
        print("✓ Model has all required properties")

        # Test statistics
        stats = model.get_sphere_statistics()
        assert "num_spheres" in stats
        assert "radius_stats" in stats
        assert "total_sphere_volume" in stats
        print("✓ Model statistics work correctly")

        return True

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")

    try:
        from morphit import MorphIt
        from losses import MorphItLosses
        from config import get_config

        # Create small model for testing
        config = get_config("MorphIt-B")
        config.model.num_spheres = 3
        config.model.num_inside_samples = 100
        config.model.num_surface_samples = 50

        model = MorphIt(config)
        losses = MorphItLosses(model)

        # Test individual loss functions
        coverage_loss = losses.compute_coverage_loss()
        overlap_penalty = losses.compute_overlap_penalty()
        boundary_penalty = losses.compute_boundary_penalty()
        surface_loss = losses.compute_surface_loss()
        containment_loss = losses.compute_containment_loss()
        sqem_loss = losses.compute_sqem_loss()

        print("✓ All individual loss functions work")

        # Test combined losses
        all_losses = losses.compute_all_losses()
        assert len(all_losses) == 6
        print("✓ Combined loss computation works")

        # Test weighted total loss
        weights = losses.get_loss_weights_from_config(config.training)
        total_loss = losses.compute_weighted_total_loss(weights)
        print("✓ Weighted total loss computation works")

        return True

    except Exception as e:
        print(f"✗ Loss computation test failed: {e}")
        traceback.print_exc()
        return False


def test_minimal_training():
    """Test minimal training run."""
    print("\nTesting minimal training...")

    try:
        from morphit import MorphIt
        from config import get_config

        # Create minimal config for quick testing
        config = get_config("MorphIt-B")
        config.model.num_spheres = 3
        config.model.num_inside_samples = 100
        config.model.num_surface_samples = 50
        config.training.iterations = 5
        config.training.verbose_frequency = 1
        config.training.density_control_min_interval = 10
        config.training.early_stopping = False
        config.visualization.off_screen = True
        config.visualization.save_video = False

        model = MorphIt(config)

        # Run minimal training
        tracker = model.train()

        print("✓ Minimal training completed successfully")

        # Test tracker save
        tracker.save()
        print("✓ Tracker save works correctly")

        # Test model save
        model.save_results("test_results.json")
        print("✓ Model save works correctly")

        return True

    except Exception as e:
        print(f"✗ Minimal training test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "config.py",
        "morphit.py",
        "losses.py",
        "density_control.py",
        "training.py",
        "convergence_tracker.py",
        "logger.py",
        "visualization.py",
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running MorphIt System Tests")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Model Creation", test_basic_model_creation),
        ("Loss Computation", test_loss_computation),
        ("Minimal Training", test_minimal_training),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
