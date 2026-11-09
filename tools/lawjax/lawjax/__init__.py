"""lawjax: the field engine's laws in JAX.

The C++ compiler lowers a law to an SSA graph (src/field/ir/LawIR.hpp) and writes it as JSON (`lawc --ir-json`). This
package builds the same step from that graph with jax.numpy, so a law can be differentiated (jax.grad through many
steps), batched (vmap) and trained against. It is the research side; the engine runs the same law in CUDA.

    from lawjax import load_law
    law = load_law("front.json")
    state = law.initial((1, 32, 128), h=0.5)
    state = law.step(state, dt=0.1)
"""

from .ir import Law, load_law
from .nets import Net, read_net, write_net

__all__ = ["Law", "load_law", "Net", "read_net", "write_net"]
