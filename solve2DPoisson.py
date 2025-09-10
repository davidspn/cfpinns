import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
from flax.training import train_state
import optax
import os
from tqdm import tqdm

#Import the fractional laplacian operator
from frac_laplacian_autodiff import compute_general_laplacian
from dynamic_caputo_full import compute_caputo_full # Still needed for time derivative

# --- 1. Define the problem & PINN structure ---

@jax.jit
def mittag_leffler_E12(t, num_terms=20):
    k = jnp.arange(num_terms)
    terms = (t**k) / jnp.exp(gammaln(k + 2))
    return jnp.sum(terms)

def u_analytical(xt: jnp.ndarray, alpha: float) -> float:
    x, t = xt[:-1], xt[-1]
    norm_sq = jnp.sum(x**2)
    spatial_part = jnp.maximum(0, 1 - norm_sq)**(1 + alpha / 2.0)
    return spatial_part * jnp.exp(-t)

def f_analytical_rhs(xt: jnp.ndarray, alpha: float, gamma: float, c: float, d: int) -> float:
    x, t = xt[:-1], xt[-1]
    norm_sq = jnp.sum(x**2)
    term1 = -t**(1 - gamma) * mittag_leffler_E12(-t) * (1 - norm_sq)**(1 + alpha/2.0)
    gamma_term1 = jnp.exp(gammaln(alpha / 2.0 + 2.0))
    gamma_term2 = jnp.exp(gammaln((d + alpha) / 2.0))
    term2_factor = c * (2**alpha) * gamma_term1 * gamma_term2
    term2_spatial = (1 - (1 + alpha / d) * norm_sq)
    term2 = term2_factor * term2_spatial * jnp.exp(-t)
    return term1 + term2

class MLP(nn.Module):
    features: list
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def u_pinn(model_apply_fn, mlp_params, xt: jnp.ndarray):
    x, t = xt[:-1], xt[-1]
    mlp_output = model_apply_fn({'params': mlp_params}, xt).squeeze()
    return (1.0 - jnp.sum(x**2)) * mlp_output

# --- 2. Define the joint Loss Function ---
def unified_pde_loss(params, batch, model_apply_fn, true_alpha, true_gamma, c, d, num_directions, learn_alpha: bool, learn_gamma: bool):
    points = batch['points']
    key = batch['key']
    
    if learn_alpha or learn_gamma:
        mlp_params = params['mlp']
        alpha_for_laplacian = nn.sigmoid(params['alpha_raw']) + 1.0 if learn_alpha else true_alpha
        gamma_for_derivative = nn.sigmoid(params['gamma_raw']) if learn_gamma else true_gamma
    else:
        mlp_params = params
        alpha_for_laplacian = true_alpha
        gamma_for_derivative = true_gamma

    u_pinn_func = partial(u_pinn, model_apply_fn, mlp_params)

    def pde_residual(xt_point, single_key):
        x_point, t_point = xt_point[:-1], xt_point[-1]
        u_time_slice = lambda t_slice: u_pinn_func(jnp.concatenate([x_point, jnp.array([t_slice])]))
        time_deriv = compute_caputo_full(f=u_time_slice, t=jnp.array([t_point]), a=0.0, alpha=gamma_for_derivative, max_n=1).squeeze()
        spatial_deriv = c * compute_general_laplacian(
            u_func=lambda x_slice, t_slice: u_pinn_func(jnp.concatenate([x_slice, jnp.array([t_slice])])),
            x=x_point, t=t_point, alpha=alpha_for_laplacian, key=single_key, d=d, num_directions=num_directions)
        lhs = time_deriv + spatial_deriv
        rhs = f_analytical_rhs(xt_point, true_alpha, true_gamma, c, d)
        return lhs - rhs

    keys = jax.random.split(key, points.shape[0])
    residuals = jax.vmap(pde_residual)(points, keys)
    loss_physics = jnp.mean(residuals**2)

    if learn_alpha or learn_gamma:
        data_points = batch['data_points']
        data_values = batch['data_values']
        data_loss_weight = batch['data_loss_weight']
        u_pred_at_data = jax.vmap(u_pinn_func)(data_points)
        loss_data = jnp.mean((u_pred_at_data - data_values)**2)
        return loss_physics + data_loss_weight * loss_data
    else:
        return loss_physics

# --- 3. Training Step ---
@partial(jax.jit, static_argnames=['model_apply_fn', 'true_alpha', 'true_gamma', 'c', 'd', 'num_directions', 'learn_alpha', 'learn_gamma'])
def train_step(state, batch, model_apply_fn, true_alpha, true_gamma, c, d, num_directions, learn_alpha: bool, learn_gamma: bool):
    loss_for_grad = partial(unified_pde_loss, model_apply_fn=model_apply_fn, true_alpha=true_alpha, true_gamma=true_gamma, c=c, d=d, num_directions=num_directions, learn_alpha=learn_alpha, learn_gamma=learn_gamma)
    loss_value, grads = jax.value_and_grad(loss_for_grad)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss_value

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- Mode Toggles ---
    SOLVE_INVERSE_PROBLEM = True
    LEARN_ALPHA = True
    LEARN_GAMMA = False

    # --- Hyperparameters ---
    TRUE_ALPHA = 1.5
    TRUE_GAMMA = 0.8
    C_CONST = 1.0
    DIMENSION = 2
    LEARNING_RATE = 1e-2
    EPOCHS = 1000
    DATA_LOSS_WEIGHT = 1000.0
    PHYSICS_BATCH_SIZE = 128
    NUM_DIRECTIONS = 16

    # --- Setup ---
    key = jax.random.PRNGKey(42)
    is_inverse = SOLVE_INVERSE_PROBLEM and (LEARN_ALPHA or LEARN_GAMMA)
    output_folder = 'plots_inverse' if is_inverse else 'plots_forward'
    os.makedirs(output_folder, exist_ok=True)
    model = MLP(features=[64, 64, 64, 64,64,64,64,1])

    # --- Conditional Initialization ---
    if is_inverse:
        print("--- Running in INVERSE mode ---")
        if LEARN_ALPHA: print("    - Learning alpha")
        if LEARN_GAMMA: print("    - Learning gamma")
        key, mlp_key = jax.random.split(key)
        mlp_params = model.init(mlp_key, jnp.zeros(DIMENSION + 1))['params']
        params = {'mlp': mlp_params}
        if LEARN_ALPHA:
            key, alpha_key = jax.random.split(key)
            params['alpha_raw'] = jax.random.normal(alpha_key, ())
        if LEARN_GAMMA:
            key, gamma_key = jax.random.split(key)
            params['gamma_raw'] = jax.random.normal(gamma_key, ())
        spatial_grid_size = 8; time_grid_size = 5
        spatial_coords = jnp.linspace(-0.95, 0.95, spatial_grid_size)
        time_coords = jnp.linspace(0.01, 0.99, time_grid_size)
        grid_x, grid_y, grid_t = jnp.meshgrid(spatial_coords, spatial_coords, time_coords)
        all_points = jnp.vstack([grid_x.ravel(), grid_y.ravel(), grid_t.ravel()]).T
        norms = jnp.linalg.norm(all_points[:, :-1], axis=1)
        data_points = all_points[norms < 1.0]
        data_values = jax.vmap(u_analytical, in_axes=(0, None))(data_points, TRUE_ALPHA)
        print(f"Using {data_points.shape[0]} data points for training.")
    else:
        print("--- Running in FORWARD mode ---")
        data_points, data_values = None, None
        key, mlp_key = jax.random.split(key)
        params = model.init(mlp_key, jnp.zeros(DIMENSION + 1))['params']

    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # --- L2 Error Calculation Setup ---
    error_grid_size = 50; err_sp_coords = jnp.linspace(-1, 1, error_grid_size)
    err_t_coords = jnp.linspace(0, 1, 10)
    err_grid_x, err_grid_y, err_grid_t = jnp.meshgrid(err_sp_coords, err_sp_coords, err_t_coords)
    all_err_points = jnp.vstack([err_grid_x.ravel(), err_grid_y.ravel(), err_grid_t.ravel()]).T
    err_norms = jnp.linalg.norm(all_err_points[:, :-1], axis=1)
    error_eval_points = all_err_points[err_norms <= 1.0]
    u_true_on_grid = jax.vmap(u_analytical, in_axes=(0, None))(error_eval_points, TRUE_ALPHA)
    norm_u_true = jnp.linalg.norm(u_true_on_grid)
    @jax.jit
    def calculate_l2_error(p_mlp):
        u_pinn_eval_func = partial(u_pinn, state.apply_fn, p_mlp)
        u_pred_on_grid = jax.vmap(u_pinn_eval_func)(error_eval_points)
        return jnp.linalg.norm(u_true_on_grid - u_pred_on_grid) / norm_u_true
    
    # --- Training Loop ---
    alpha_history, gamma_history, l2_error_history = [], [], []
    alpha_for_error_plot, gamma_for_error_plot = [], []
    pbar = tqdm(range(EPOCHS), desc=f"PINN Training ({'Inverse' if is_inverse else 'Forward'})")
    
    for epoch in pbar:
        key, pkey, lkey = jax.random.split(key, 3)
        r = jnp.sqrt(jax.random.uniform(pkey, (PHYSICS_BATCH_SIZE,)))
        phi = jax.random.uniform(pkey, (PHYSICS_BATCH_SIZE,)) * 2.0 * jnp.pi
        phys_x = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi)], axis=1)
        phys_t = jax.random.uniform(pkey, (PHYSICS_BATCH_SIZE, 1)) * 0.99 + 0.01
        physics_points = jnp.hstack([phys_x, phys_t])

        batch = {'points': physics_points, 'key': lkey}
        if is_inverse:
            batch.update({'data_points': data_points, 'data_values': data_values, 'data_loss_weight': DATA_LOSS_WEIGHT})
        
        state, loss = train_step(state, batch, model.apply, TRUE_ALPHA, TRUE_GAMMA, C_CONST, DIMENSION, NUM_DIRECTIONS, learn_alpha=LEARN_ALPHA and is_inverse, learn_gamma=LEARN_GAMMA and is_inverse)
        
        log_data = {"loss": f"{loss:.4e}"}
        mlp_params_for_error = state.params['mlp'] if is_inverse else state.params
        
        if LEARN_ALPHA and is_inverse:
            current_alpha = nn.sigmoid(state.params.get('alpha_raw', 0.0)) + 1.0
            alpha_history.append(current_alpha)
            log_data["L_α"] = f"{current_alpha:.3f}"
        else:
            current_alpha = TRUE_ALPHA

        if LEARN_GAMMA and is_inverse:
            current_gamma = nn.sigmoid(state.params.get('gamma_raw', 0.0))
            gamma_history.append(current_gamma)
            log_data["L_γ"] = f"{current_gamma:.3f}"
        else:
            current_gamma = TRUE_GAMMA

        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            error_val = calculate_l2_error(mlp_params_for_error)
            l2_error_history.append(error_val)
            alpha_for_error_plot.append(current_alpha)
            gamma_for_error_plot.append(current_gamma)
            log_data["L2_err"] = f"{error_val:.4f}"

        if epoch % 100 == 0:
            pbar.set_postfix(log_data)
    
    # --- Visualization ---
    if is_inverse:
        final_alpha = alpha_history[-1] if LEARN_ALPHA else TRUE_ALPHA
        final_gamma = gamma_history[-1] if LEARN_GAMMA else TRUE_GAMMA
    else:
        final_alpha = TRUE_ALPHA
        final_gamma = TRUE_GAMMA

    if is_inverse:
        # (Plotting for alpha convergence)
        if LEARN_ALPHA:
            print(f"\nTrue Alpha: {TRUE_ALPHA}, Final Learned Alpha: {final_alpha:.6f}")
            plt.figure(figsize=(10, 6)); plt.plot(alpha_history, label='Learned Alpha')
            plt.axhline(y=TRUE_ALPHA, color='r', linestyle='--', label=f'True Alpha = {TRUE_ALPHA}')
            plt.xlabel('Epoch'); plt.ylabel('Alpha Value'); plt.legend(); plt.grid(True)
            plt.ylim(1.0, 2.0); plt.title('Convergence of Learned Alpha')
            plt.savefig(f'{output_folder}/alpha_convergence.png', dpi=300); plt.show()
        # (Plotting for gamma convergence)
        if LEARN_GAMMA:
            print(f"True Gamma: {TRUE_GAMMA}, Final Learned Gamma: {final_gamma:.6f}")
            plt.figure(figsize=(10, 6)); plt.plot(gamma_history, label='Learned Gamma')
            plt.axhline(y=TRUE_GAMMA, color='r', linestyle='--', label=f'True Gamma = {TRUE_GAMMA}')
            plt.xlabel('Epoch'); plt.ylabel('Gamma Value'); plt.legend(); plt.grid(True)
            plt.ylim(0.0, 1.0); plt.title('Convergence of Learned Gamma')
            plt.savefig(f'{output_folder}/gamma_convergence.png', dpi=300); plt.show()
        
        # (L2 Error vs Parameters plot)
        fig, ax1 = plt.subplots(figsize=(12, 7))
        fig.suptitle('L2 Error of Solution vs. Learned Parameters during Training')
        ax1.set_ylabel('L2 Relative Error (log scale)'); ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="--")
        if LEARN_ALPHA:
            color = 'tab:blue'
            ax1.set_xlabel('Learned Alpha ($\\alpha$)', color=color)
            ax1.plot(alpha_for_error_plot, l2_error_history, '.-', color=color)
            ax1.tick_params(axis='x', labelcolor=color)
            ax1.axvline(x=TRUE_ALPHA, color=color, linestyle='--', label=f'True Alpha = {TRUE_ALPHA:.2f}')
        if LEARN_GAMMA:
            ax2 = ax1.twiny()
            color = 'tab:green'
            ax2.set_xlabel('Learned Gamma ($\\gamma$)', color=color)
            ax2.plot(gamma_for_error_plot, l2_error_history, '.-', color=color, alpha=0.6)
            ax2.tick_params(axis='x', labelcolor=color)
            ax2.axvline(x=TRUE_GAMMA, color=color, linestyle='--', label=f'True Gamma = {TRUE_GAMMA:.2f}')
        ax1.scatter([alpha_for_error_plot[-1]], [l2_error_history[-1]], color='red', s=120, zorder=5, label='Final Value')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{output_folder}/l2_error_vs_params.png', dpi=300); plt.show()

# Solution plots at t=0.5
print("Visualizing final solution at t=0.5...")
t_slice = 0.5
vis_grid_size = 100
vis_coords = jnp.linspace(-1, 1, vis_grid_size)
grid_x_vis, grid_y_vis = jnp.meshgrid(vis_coords, vis_coords)
mask = grid_x_vis**2 + grid_y_vis**2 > 1
x_flat = grid_x_vis.ravel(); y_flat = grid_y_vis.ravel()
t_flat = jnp.full_like(x_flat, t_slice)
eval_points = jnp.vstack([x_flat, y_flat, t_flat]).T

mlp_p = state.params['mlp'] if LEARN_ALPHA else state.params
u_pinn_final_func = partial(u_pinn, state.apply_fn, mlp_p)
u_pred_flat = jax.vmap(u_pinn_final_func)(eval_points)
u_pred = u_pred_flat.reshape(vis_grid_size, vis_grid_size)
u_true_flat = jax.vmap(u_analytical, in_axes=(0, None))(eval_points, TRUE_ALPHA)
u_true = u_true_flat.reshape(vis_grid_size, vis_grid_size)

u_pred = jnp.where(mask, jnp.nan, u_pred); u_true = jnp.where(mask, jnp.nan, u_true)
error = jnp.abs(u_pred - u_true)

# --- Plot 1: PINN Solution ---
fig1, ax1 = plt.subplots(figsize=(8, 7))
title_alpha_str = f'Learned $\\alpha={final_alpha:.3f}$' if LEARN_ALPHA else f'Fixed $\\alpha={TRUE_ALPHA}$'
im1 = ax1.imshow(u_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
ax1.set_title(f'PINN Solution at t={t_slice} ({title_alpha_str})')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
fig1.colorbar(im1, ax=ax1)
plt.tight_layout()
plt.savefig(f'{output_folder}/pinn_solution.png', dpi=300)
plt.show()

# --- Plot 2: True Solution ---
fig2, ax2 = plt.subplots(figsize=(8, 7))
im2 = ax2.imshow(u_true, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
ax2.set_title(f'True Solution at t={t_slice} ($\\alpha={TRUE_ALPHA}$)')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
if LEARN_ALPHA:
    data_points_at_slice = data_points[jnp.abs(data_points[:, 2] - t_slice) < 0.15]
fig2.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.savefig(f'{output_folder}/true_solution.png', dpi=300)
plt.show()

# --- Plot 3: Absolute Error ---
fig3, ax3 = plt.subplots(figsize=(8, 7))
im3 = ax3.imshow(error, extent=[-1, 1, -1, 1], origin='lower', cmap='cividis')
ax3.set_title('Absolute Error $|u_{true} - u_{pinn}|$ at t=' + str(t_slice))
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
fig3.colorbar(im3, ax=ax3)
plt.tight_layout()
plt.savefig(f'{output_folder}/absolute_error.png', dpi=300)
plt.show()