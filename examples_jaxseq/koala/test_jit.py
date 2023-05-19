import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from functools import partial
from jax.experimental.pjit import pjit
import numpy as np

mesh = Mesh(mesh_utils.create_device_mesh((2, 16)), axis_names=('dp', 'mp'))

print(mesh)

# prng = jax.random.PRNGKey(0)
# prng, _ = jax.random.split(prng)
# print(jax.random.randint(prng, (2, 16), 0, 10))
# print(prng)
# print(np.asarray(prng))
# print(np.asarray(jax.random.randint(prng, (2, 16), 0, 10)))

@partial(
    pjit, 
    donate_argnums=(0,), 
    in_shardings=NamedSharding(mesh, PS()), 
    out_shardings=NamedSharding(mesh, PS()), 
)
def identity_fn1(x):
    # x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PS()))
    # jax.debug.print("{x}", x=x)
    # jax.debug.print("{mean} +- {std}", mean=x.mean(), std=x.std())
    jax.debug.visualize_array_sharding(x)
    return x

# @partial(
#     pjit, 
#     donate_argnums=(0,), 
#     # in_shardings=NamedSharding(mesh, PS('dp')), 
#     out_shardings=NamedSharding(mesh, PS()), 
# )
# def identity_fn2(x):
#     # x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PS()))
#     jax.debug.print("{x}", x=x)
#     jax.debug.print("{mean} +- {std}", mean=x.mean(), std=x.std())
#     return x

# with mesh:

with jax.default_device(jax.devices('cpu')[0]):
    x = jnp.full((4, 16), 0, dtype=jnp.float32)
#     if jax.process_index() == 0:
#         x = jnp.full((4, 16), 0, dtype=jnp.float32)
#     else:
#         x = jnp.full((4, 16), 1, dtype=jnp.float32)



# print('Before:')
# jax.debug.visualize_array_sharding(x)
# print('Before:', jax.process_index(), x.shape, x)

# multihost_utils.assert_equal(x)

x = identity_fn1(x)
# x = multihost_utils.broadcast_one_to_all(x)

# print('After:')
# jax.debug.visualize_array_sharding(x)
# print('After:', jax.process_index(), x.shape, x)
# print(x.mean())
# print('PART 1:', jax.device_get(identity_fn2(x)))
# print('PART 2:', identity_fn2(x))
# jax.debug.visualize_array_sharding(identity_fn2(x))
# print('PART 2:', identity_fn2(jax.device_get(identity_fn2(x))))

# @partial(
#     jax.jit, 
#     donate_argnums=(0,), 
#     in_shardings=(NamedSharding(mesh, PS()),), 
#     out_shardings=NamedSharding(mesh, PS()), 
# )
# def identity_fn(x):
#     # x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PS()))
#     jax.debug.print("{mean} +- {std}", mean=x.mean(), std=x.std())
#     return x

# with jax.default_device(jax.devices('cpu')[0]):
#     x = jnp.full((2**14, 2**14), 0.0, dtype=jnp.float32)

# print('Before:')
# jax.debug.visualize_array_sharding(x)
# print('Before:', jax.process_index(), x.shape, x.mean(), x.std())

# x = multihost_utils.host_local_array_to_global_array(x, mesh, PS())

# print('After1:')
# jax.debug.visualize_array_sharding(x)
# print('After1:', jax.process_index(), x.shape)

# x = identity_fn(x)

# print('After2:')
# jax.debug.visualize_array_sharding(x)
# print('After2:', jax.process_index(), x.shape)
