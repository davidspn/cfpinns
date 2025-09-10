import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
from flax.training import train_state
import optax

# Import from our framework
from cfpinns_framework.pinn_framework import MLP, create_pinn_state, train_step
from cfpinns_framework.dynamic_caputo_operator import compute_caputo_0_to_1

# --- 1. Define the Problem-Specific Loss Function for the Inverse Problem ---
def inverse_loss_fn(apply_fn, params, batch, a):
    collocation_points = batch['physics_points']
    data_x, data_y = batch['data_x'], batch['data_y']
    data_loss_weight = batch['data_loss_weight']
    
    mlp_params = params['mlp']
    alpha_raw = params['alpha_raw']
    alpha = nn.sigmoid(alpha_raw)

    def y_pinn(x):
        return x * apply_fn({'params': mlp_params}, x)
        
    y_op_pred = compute_caputo_0_to_1(f=y_pinn, t=collocation_points, a=a, alpha=alpha)
    gamma_val = jnp.exp(gammaln(2.5))
    y_op_true = (2.0 / gamma_val) * collocation_points**1.5
    loss_physics = jnp.mean((y_op_pred - y_op_true)**2)

    y_pred_at_data = jax.vmap(y_pinn)(data_x)
    loss_data = jnp.mean((y_pred_at_data - data_y)**2)
    
    total_loss = loss_physics + data_loss_weight * loss_data
    return total_loss

# --- 2. Define the Analytical Solutions for Final Plotting ---
def solution_caputo(x):
    return x**2

def solution_conformable(x):
    gamma_val = jnp.exp(gammaln(2.5))
    return (1.0 / gamma_val) * x**2

def solution_improved_conformable(x):
    gamma_val = jnp.exp(gammaln(2.5))
    term1_inner = x**1.5 - 1.5*x + 1.5*jnp.sqrt(x) - 0.75
    term1 = (4.0 / gamma_val) * term1_inner
    term2 = (3.0 / gamma_val) * jnp.exp(-2 * jnp.sqrt(x))
    return term1 + term2

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    key = jax.random.PRNGKey(42)
    learning_rate = 1e-3
    epochs = 10000
    A_PARAM = 0.0
    TRUE_ALPHA = 0.5
    
    # --- Generate Data ---
    N_data_points = 10
    data_x = jnp.linspace(1e-6, 1.0, N_data_points, dtype=jnp.float32)
    # The "observed" data comes from the improved conformable solution
    data_y = solution_improved_conformable(data_x)
    
    N_physics_points = 100
    physics_points = jnp.linspace(1e-6, 1.0, N_physics_points, dtype=jnp.float32)
    
    batch_data = {
        'physics_points': physics_points,
        'data_x': data_x,
        'data_y': data_y,
        'data_loss_weight': 100.0
    }

    # --- Model and State Initialization ---
    model = MLP(features=[32, 32, 1])
    mlp_params = model.init(key, jnp.zeros(1))['params']
    key, subkey = jax.random.split(key)
    alpha_raw_param = jax.random.normal(subkey, ())
    params = {'mlp': mlp_params, 'alpha_raw': alpha_raw_param}
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # --- Prepare for Training ---
    loss_for_training = partial(inverse_loss_fn, a=A_PARAM)

    # --- Training Loop ---
    print(f"Starting inverse problem training. True alpha = {TRUE_ALPHA}")
    alpha_history = []
    for epoch in range(epochs):
        state, loss = train_step(state, batch_data, loss_function=loss_for_training)
        current_alpha = nn.sigmoid(state.params['alpha_raw'])
        alpha_history.append(current_alpha)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, Learned Alpha: {current_alpha:.6f}")
    print("Training finished.")

    # --- Results and Visualization ---
    final_alpha = nn.sigmoid(state.params['alpha_raw'])
    print(f"\nTrue Alpha: {TRUE_ALPHA}")
    print(f"Final Learned Alpha: {final_alpha:.6f}")

    # Plot 1: Convergence of alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_history, label='Learned Alpha')
    plt.axhline(y=TRUE_ALPHA, color='r', linestyle='--', label=f'True Alpha = {TRUE_ALPHA}')
    plt.xlabel('Epoch'); plt.ylabel('Alpha Value'); plt.legend(); plt.grid(True)
    plt.title('Convergence of Learned Fractional Order (alpha)')
    plt.show()

    # Plot 2: Comparison of the final PINN solution to the analytical solutions
    def y_pinn_solution(p_mlp, x):
        return x * state.apply_fn({'params': p_mlp}, x)

    plot_points = jnp.linspace(0, 1.0, 400, dtype=jnp.float32)
    # Use the final trained MLP parameters to get the prediction
    final_mlp_params = state.params['mlp']
    y_pred = jax.vmap(lambda x: y_pinn_solution(final_mlp_params, x))(plot_points)
    
    y_true_caputo = solution_caputo(plot_points)
    y_true_conformable = solution_conformable(plot_points)
    y_true_improved = solution_improved_conformable(plot_points)
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_points, y_true_caputo, 'r-', label='Exact Caputo Solution', linewidth=2)
    plt.plot(plot_points, y_true_conformable, 'g:', label='Exact Conformable Solution', linewidth=2)
    plt.plot(plot_points, y_true_improved, 'k-', label='Exact Improved Conformable', linewidth=3)
    plt.plot(plot_points, y_pred, 'b--', label=f'PINN Solution (Learned α={final_alpha:.3f})', linewidth=3)
    # Also plot the given data points to show what the model was trained on
    plt.plot(data_x, data_y, 'ro', markersize=8, label='Given Data Points')
    
    plt.xlabel('x'); plt.ylabel('y(x)'); plt.legend(); plt.grid(True)
    plt.ylim(bottom=-0.1); plt.xlim(0, 1)
    plt.title('Final PINN Solution vs. Analytical Definitions')
    plt.show()