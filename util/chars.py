import torch, random
import numpy as np

chars = " 0123456789-,."
nchars = len(chars)
idx = {}
for i, c in enumerate(chars): idx[c] = i

def input_to_string(inp):
    """Converts normalized logits to a string just by picking a random symbol
    according to the distribution of the logits for each time sequence.
    This kind of helps understand how training is going.

    DO NOT USE for prediction. Use the beam search as in beamtst.ipynb"""
    assert inp.shape[0] == 1
    if isinstance(inp, torch.Tensor):
        inp = inp.clone().detach().cpu().numpy()
    def randchoice(p):
        p = np.exp(p)
        u = random.random()
        #print(p)
        for i in range(p.shape[0]):
            u -= p[i]
            if u < 1e-6:
                return i
        raise Exception("not a probability distribution")
    return "".join(chars[randchoice(inp[0, j, :])] for j in range(inp.shape[1]))
  
