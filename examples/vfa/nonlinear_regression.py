"""Vector Function Architecture - Nonlinear Regression Example.

This example demonstrates nonlinear function regression using VFA, reproducing
concepts from Frady et al. 2021 §7.2.2.

Based on:
    Frady et al. 2021: "Computing on Functions Using Randomized Vector
    Representations"
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vsax import VSAMemory, create_fhrr_model
from vsax.vfa import NonlinearRegressor


def main():
    """Demonstrate nonlinear regression with VFA."""
    print("=" * 70)
    print("Vector Function Architecture - Nonlinear Regression")
    print("=" * 70)

    # Initialize FHRR model with high dimensionality for better approximation
    key = jax.random.PRNGKey(42)
    model = create_fhrr_model(dim=2048, key=key)

    print("\n1. Test Function 1: Sine Wave with Noise")
    print("-" * 70)

    # Generate noisy sine data
    x_train = jnp.linspace(0, 2 * jnp.pi, 40)
    y_true = jnp.sin(x_train)
    noise = jax.random.normal(key, x_train.shape) * 0.1
    y_train = y_true + noise

    print(f"  Training samples: {len(x_train)}")
    print(f"  Noise level: sigma=0.1")

    # Fit regressor
    memory1 = VSAMemory(model)
    regressor1 = NonlinearRegressor(model, memory1, regularization=1e-6)
    regressor1.fit(x_train, y_train)

    # Predict on fine grid
    x_test = jnp.linspace(0, 2 * jnp.pi, 200)
    y_pred = regressor1.predict(x_test)
    y_test_true = jnp.sin(x_test)

    # Compute R² score
    r2 = regressor1.score(x_test, y_test_true)
    print(f"  R² score on test set: {r2:.4f}")

    print("\n2. Test Function 2: Damped Oscillation")
    print("-" * 70)

    # Generate damped oscillation data
    x_train2 = jnp.linspace(0, 10, 50)
    y_train2 = jnp.exp(-0.3 * x_train2) * jnp.cos(2 * x_train2)

    print(f"  Training samples: {len(x_train2)}")
    print(f"  Function: f(x) = exp(-0.3x) * cos(2x)")

    # Fit regressor
    memory2 = VSAMemory(model)
    regressor2 = NonlinearRegressor(model, memory2, regularization=1e-6)
    regressor2.fit(x_train2, y_train2)

    # Predict
    x_test2 = jnp.linspace(0, 10, 200)
    y_pred2 = regressor2.predict(x_test2)
    y_test_true2 = jnp.exp(-0.3 * x_test2) * jnp.cos(2 * x_test2)

    r2_2 = regressor2.score(x_test2, y_test_true2)
    print(f"  R² score on test set: {r2_2:.4f}")

    print("\n3. Test Function 3: Polynomial with Sharp Features")
    print("-" * 70)

    # Generate polynomial data
    x_train3 = jnp.linspace(-2, 2, 30)
    y_train3 = 0.5 * x_train3**3 - 2 * x_train3 + 1

    print(f"  Training samples: {len(x_train3)}")
    print(f"  Function: f(x) = 0.5x³ - 2x + 1")

    # Fit regressor
    memory3 = VSAMemory(model)
    regressor3 = NonlinearRegressor(model, memory3, regularization=1e-6)
    regressor3.fit(x_train3, y_train3)

    # Predict
    x_test3 = jnp.linspace(-2, 2, 200)
    y_pred3 = regressor3.predict(x_test3)
    y_test_true3 = 0.5 * x_test3**3 - 2 * x_test3 + 1

    r2_3 = regressor3.score(x_test3, y_test_true3)
    print(f"  R² score on test set: {r2_3:.4f}")

    print("\n4. Comparing Different Regularization Strengths")
    print("-" * 70)

    # Test effect of regularization on sine function
    regularizations = [1e-8, 1e-6, 1e-4, 1e-2]
    reg_results = {}

    for reg in regularizations:
        memory_reg = VSAMemory(model)
        reg_model = NonlinearRegressor(model, memory_reg, regularization=reg)
        reg_model.fit(x_train, y_train)
        y_pred_reg = reg_model.predict(x_test)
        r2_reg = reg_model.score(x_test, y_test_true)
        reg_results[reg] = (y_pred_reg, r2_reg)
        print(f"  lambda={reg:8.0e}: R² = {r2_reg:.4f}")

    print("\n5. Visualization")
    print("-" * 70)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Sine wave
    ax = axes[0, 0]
    ax.scatter(x_train, y_train, c="gray", s=30, alpha=0.6, label="Training data")
    ax.plot(x_test, y_test_true, "k--", linewidth=2, label="True function")
    ax.plot(x_test, y_pred, "r-", linewidth=2, label="VFA prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Sine Wave (R² = {r2:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Damped oscillation
    ax = axes[0, 1]
    ax.scatter(x_train2, y_train2, c="gray", s=30, alpha=0.6, label="Training data")
    ax.plot(x_test2, y_test_true2, "k--", linewidth=2, label="True function")
    ax.plot(x_test2, y_pred2, "r-", linewidth=2, label="VFA prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Damped Oscillation (R² = {r2_2:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Polynomial
    ax = axes[0, 2]
    ax.scatter(x_train3, y_train3, c="gray", s=30, alpha=0.6, label="Training data")
    ax.plot(x_test3, y_test_true3, "k--", linewidth=2, label="True function")
    ax.plot(x_test3, y_pred3, "r-", linewidth=2, label="VFA prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Polynomial (R² = {r2_3:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residuals for sine
    ax = axes[1, 0]
    residuals = y_pred - y_test_true
    ax.plot(x_test, residuals, linewidth=2, color="red")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.fill_between(x_test, residuals, alpha=0.3, color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("Residual")
    ax.set_title("Prediction Errors (Sine)")
    ax.grid(True, alpha=0.3)

    # Plot 5: Effect of regularization
    ax = axes[1, 1]
    ax.plot(x_test, y_test_true, "k--", linewidth=2, label="True")
    for reg in regularizations:
        y_pred_reg, r2_reg = reg_results[reg]
        ax.plot(x_test, y_pred_reg, linewidth=2, alpha=0.7, label=f"lambda={reg:.0e}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Effect of Regularization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: R² vs regularization
    ax = axes[1, 2]
    reg_values = list(regularizations)
    r2_values = [reg_results[reg][1] for reg in reg_values]
    ax.semilogx(reg_values, r2_values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Regularization (lambda)")
    ax.set_ylabel("R^2 Score")
    ax.set_title("Model Performance vs Regularization")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vfa_nonlinear_regression.png", dpi=150, bbox_inches="tight")
    print("  [OK] Plots saved to vfa_nonlinear_regression.png")

    print("\n6. Function Arithmetic")
    print("-" * 70)

    # Demonstrate function addition
    x_arith = jnp.linspace(0, 2 * jnp.pi, 30)
    y_sin = jnp.sin(x_arith)
    y_cos = jnp.cos(x_arith)

    memory_sin = VSAMemory(model)
    memory_cos = VSAMemory(model)

    reg_sin = NonlinearRegressor(model, memory_sin)
    reg_cos = NonlinearRegressor(model, memory_cos)

    reg_sin.fit(x_arith, y_sin)
    reg_cos.fit(x_arith, y_cos)

    # Add functions: sin + cos
    from vsax.vfa import VectorFunctionEncoder

    vfa = VectorFunctionEncoder(model, memory_sin)
    combined_hv = vfa.add_functions(reg_sin._function_hv, reg_cos._function_hv)

    # Evaluate combined function
    x_eval = jnp.linspace(0, 2 * jnp.pi, 100)
    y_combined = vfa.evaluate_batch(combined_hv, x_eval)
    y_expected = jnp.sin(x_eval) + jnp.cos(x_eval)

    error = jnp.mean(jnp.abs(y_combined - y_expected))
    print(f"  Computed: sin(x) + cos(x)")
    print(f"  Mean absolute error: {error:.4f}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - VFA can approximate complex nonlinear functions")
    print("  - Works well for periodic, polynomial, and composite functions")
    print("  - Regularization controls overfitting")
    print("  - Supports function arithmetic (addition, scaling)")
    print("  - Compact hypervector representation")
    print("=" * 70)


if __name__ == "__main__":
    main()
