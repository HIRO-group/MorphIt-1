"""
Simple example demonstrating MorphIt usage.
"""

from config import get_config
from morphit import MorphIt
from visualization import visualize_packing


def simple_example():
    """Simple example of using MorphIt."""
    print("=== Simple MorphIt Example ===")

    # Get default configuration
    config = get_config("MorphIt-B")

    # Customize some parameters
    config.model.num_spheres = 20
    config.model.mesh_path = "../mesh_models/fr3/collision/link0.obj"
    config.training.iterations = 500
    config.training.verbose_frequency = 5
    config.visualization.off_screen = False
    config.visualization.save_video = False

    # Create model
    model = MorphIt(config)

    # Show initial state
    print("\nInitial configuration:")
    model.print_statistics()

    # Train the model
    print("\nTraining...")
    tracker = model.train()

    # Show final state
    print("\nFinal configuration:")
    model.print_statistics()

    # Visualize result
    print("\nVisualizing result...")
    visualize_packing(model, show_sample_points=False)

    # Save results
    model.save_results("simple_example_results.json")
    tracker.save()

    print("\nExample complete!")


def quick_test():
    """Quick test with minimal parameters."""
    print("=== Quick MorphIt Test ===")

    # Create model with default config
    model = MorphIt()

    # Train with custom settings
    tracker = model.train(
        {
            "training.iterations": 20,
            "training.verbose_frequency": 5,
            "visualization.off_screen": True,
            "visualization.save_video": False,
        }
    )

    # Show results
    model.print_statistics()
    print(f"Training completed in {len(tracker.metrics['total_loss'])} iterations")

    # Quick visualization
    visualize_packing(model, show_sample_points=False, show_surface_points=False)


def compare_loss_configurations():
    """Compare different loss configurations."""
    print("=== Comparing Loss Configurations ===")

    loss_configs = ["MorphIt-V", "MorphIt-S", "MorphIt-B"]
    results = {}

    for loss_config in loss_configs:
        print(f"\nTesting {loss_config}...")

        # Create model with specific loss configuration
        config = get_config(loss_config)
        config.training.iterations = 25
        config.training.verbose_frequency = 10
        config.visualization.off_screen = True
        config.visualization.save_video = False

        model = MorphIt(config)
        tracker = model.train()

        # Store results
        final_stats = model.get_sphere_statistics()
        results[loss_config] = {
            "final_loss": tracker.metrics["total_loss"][-1],
            "sphere_count": final_stats["num_spheres"],
            "volume_ratio": final_stats["volume_ratio"],
            "iterations": len(tracker.metrics["total_loss"]),
        }

        # Save results
        model.save_results(f"{loss_config.lower()}_results.json")

    # Print comparison
    print("\n=== Comparison Results ===")
    for config_name, metrics in results.items():
        print(f"{config_name}:")
        print(f"  Final loss: {metrics['final_loss']:.6f}")
        print(f"  Sphere count: {metrics['sphere_count']}")
        print(f"  Volume ratio: {metrics['volume_ratio']:.4f}")
        print(f"  Iterations: {metrics['iterations']}")


if __name__ == "__main__":
    # Run simple example
    simple_example()

    # Uncomment to run quick test
    # quick_test()

    # Uncomment to compare loss configurations
    # compare_loss_configurations()
