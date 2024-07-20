from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import blackjax
import optax
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import powerbox as pbox
from tqdm import trange
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


def make_field(seed, A, B, data_dim):
    pb = pbox.PowerBox(
        N=data_dim,                    
        dim=1,                   
        pk=lambda k: A * k ** -B,
        boxlength=1.0,          
        seed=seed,              
        ensure_physical=True    
    )
    return pb.delta_x()


def make_fields(parameters, data_dim):
    fields = []
    for i in range(len(parameters)):
        fields.append(
            make_field(
                i, parameters[i, 0], parameters[i, 1], data_dim
            )
        )
    return jnp.stack(fields)


class MLP(nn.Module):
    hidden_dim: int = 32
    out_dim: int = 2
    n_layers: int = 3

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class AffineBijector:
    def __init__(self, shift_and_log_scale):
        self.shift_and_log_scale = shift_and_log_scale

    def forward_and_log_det(self, x):
        shift, log_scale = jnp.split(self.shift_and_log_scale, 2, axis=-1)
        y = x * jnp.exp(log_scale) + shift
        log_det = log_scale
        return y, log_det

    def inverse_and_log_det(self, y):
        shift, log_scale = jnp.split(self.shift_and_log_scale, 2, axis=-1)
        x = (y - shift) * jnp.exp(-log_scale)
        log_det = -log_scale
        return x, log_det


class MaskedCoupling:
    def __init__(self, mask, conditioner, bijector):
        """Coupling layer with masking and conditioner."""
        self.mask = mask
        self.conditioner = conditioner
        self.bijector = bijector

    def forward_and_log_det(self, x, q):
        """Transforms masked indices of `x` conditioned on unmasked indices using bijector."""
        x_cond = jnp.where(self.mask, 0., x)
        bijector_params = self.conditioner(jnp.concatenate([x_cond, q], axis=1))
        y, log_det = self.bijector(bijector_params).forward_and_log_det(x)
        log_det = jnp.where(self.mask, log_det, 0.0)
        y = jnp.where(self.mask, y, x)
        return y, jnp.sum(log_det, axis=-1)

    def inverse_and_log_det(self, y, q):
        """Transforms masked indices of `y` conditioned on unmasked indices using bijector."""
        y_cond = jnp.where(self.mask, 0., y)
        bijector_params = self.conditioner(jnp.concatenate([y_cond, q], axis=1))
        x, log_det = self.bijector(bijector_params).inverse_and_log_det(y)
        log_det = jnp.where(self.mask, log_det, 0.0)
        x = jnp.where(self.mask, x, y)
        return x, jnp.sum(log_det, axis=-1)
        

class RealNVP(nn.Module):
    n_transforms: int = 4
    d_params: int = 2
    d_q: int = 2
    d_hidden: int = 128
    n_layers: int = 4

    def setup(self):
        self.mask_list = [
            jnp.arange(self.d_params) % 2 == i % 2 
            for i in range(self.n_transforms)
        ]
        self.conditioner_list = [
            MLP(
                self.d_hidden + self.d_q, 
                2 * self.d_params, 
                self.n_layers
            ) 
            for _ in range(self.n_transforms)
        ]
        self.base_dist = tfp.distributions.Normal(
            loc=jnp.zeros(self.d_params), scale=jnp.ones(self.d_params)
        )
    
    def log_prob(self, x, q):
        log_prob = jnp.zeros(x.shape[:-1])
        for mask, conditioner in zip(
            self.mask_list[::-1], 
            self.conditioner_list[::-1]
        ):
            x, ldj = MaskedCoupling(
                mask, conditioner, AffineBijector
            ).inverse_and_log_det(x, q)
            log_prob += ldj
        return log_prob + self.base_dist.log_prob(x).sum(-1)

    def sample(self, sample_shape, key, y, n_transforms=None):
        x = self.base_dist.sample(key, sample_shape)
        for mask, conditioner in zip(
            self.mask_list[:n_transforms], 
            self.conditioner_list[:n_transforms]
        ):
            x, _ = MaskedCoupling(
                mask, conditioner, AffineBijector
            ).forward_and_log_det(x, y)
        return x

    def __call__(self, x, q):
        return self.log_prob(x, q)


if __name__ == "__main__":

    key = jr.key(0)

    # Model and training
    data_dim = 64
    parameter_dim = 2
    n_data = 50_000
    n_samples = 10
    n_steps = 50_000
    n_batch = 128
    lr = 1e-4
    # MCMC sampling
    num_samples = 10_000
    num_chains = 1
    inv_mass_matrix = jnp.ones((parameter_dim,)) * 0.1 
    step_size = 1e-2
    # Data
    lower = jnp.array([0.1, 1.])
    upper = jnp.array([1.5, 4.])

    parameter_prior = tfd.Blockwise(
        [tfd.Uniform(lower[p], upper[p]) for p in range(parameter_dim)]
    )

    key, key_q, key_init, key_sample = jr.split(key, 4)

    q = parameter_prior.sample((n_data,), seed=key_q)
    x = make_fields(q, data_dim)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    assert jnp.isfinite(x).all()

    model = RealNVP(d_params=data_dim, d_q=parameter_dim, d_hidden=256)
    params = model.init(key_init, x[:2], q[:2])

    assert jnp.isfinite(model.apply(params, x[:3], q[:3]).mean())

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)


    @jax.jit
    def train_step(params, opt_state, x, q):
        def loss_fn(params):
            return -model.apply(params, x, q).mean()
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state


    losses = []
    with trange(n_steps) as steps:
        for step in steps:

            # Draw a random batches from x
            idx = jr.choice(jr.fold_in(key, step), x.shape[0], shape=(n_batch,))
            
            x_batch, q_batch = x[idx], q[idx]

            loss, params, opt_state = train_step(
                params, opt_state, x_batch, q_batch
            )

            steps.set_postfix(val=loss)
            losses.append(loss)

    q = parameter_prior.sample((n_samples,), seed=key)
    x_sample = model.apply(params, key_sample, (n_samples,), q, method=model.sample)

    fig, axs = plt.subplots(1, 2, dpi=200)
    ax = axs[0]
    ax.plot(x[:10])
    ax = axs[1]
    ax.plot(x_sample)
    plt.savefig("flow.png")
    plt.close()

    plt.figure(dpi=200)
    plt.plot(losses)
    plt.savefig("loss.png")
    plt.close()


    def inference_loop_multiple_chains(
        rng_key, kernel, initial_state, num_samples, num_chains
    ):

        @jax.jit
        def one_step(states, rng_key):
            keys = jr.split(rng_key, num_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states

        keys = jr.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    key_q, key_sample = jr.split(key)

    # Datavector and its true parameters
    parameters = parameter_prior.sample((1,), seed=key)
    observed = make_fields(parameters, data_dim)

    def density_fn(x, observed):
        pi = x["pi"][None, :] 
        log_prob = model.apply(params, observed, pi) + parameter_prior.log_prob(pi)
        return log_prob.squeeze()


    # Initialise MCMC on posterior function (scaling data, very important)
    nuts = blackjax.nuts(
        partial(density_fn, observed=scaler.fit_transform(observed)), 
        step_size, 
        inv_mass_matrix
    )
    nuts_kernel = jax.jit(nuts.step)

    # Same initial positions for sampling
    initial_positions = {"pi" : parameters} # Cheating (this has to be (num_chains, parameter_dim) shaped)
    initial_states = jax.vmap(nuts.init)(initial_positions)

    # Sample posterior given datavector (same sampling key)
    states = inference_loop_multiple_chains(
        key_sample, 
        jax.jit(nuts.step), 
        initial_states, 
        num_samples, 
        num_chains 
    )

    print("True parameters", parameters)

    plt.figure(dpi=200)
    plt.hist2d(*states.position["pi"].squeeze().T, bins=100, cmap="PuOr")
    plt.scatter(*parameters.T, color="r", marker="x")
    plt.xlabel("A")
    plt.ylabel("B")
    plt.savefig("mcmc.png")
    plt.close()