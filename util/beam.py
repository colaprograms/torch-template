"""
A beam search implementation.

All probabilities in this module are logs.
"""
import torch
import numpy as np
from util.chars import chars, nchars, idx, input_to_string

def lae(*z):
    z = np.array(z)
    m = np.max(z)
    return np.log(np.sum(np.exp(z - m))) + m if m != -np.inf else -np.inf

def prefix_to_string(pr):
    return "".join([chars[z] for z in pr])

class beam:
    def __init__(self, prefix, p):
        """Make a beam

        prefix:
            the prefix, a tuple of integers.
        lb_p, lnb_p:
            log of the estimated probability that this prefix was
            correct up to the current time and the last symbol
            was (blank, nonblank)
        """
        self.prefix = prefix
        self.p = p

    def __repr__(self):
        return "beam(%s, %.4f)" % (prefix_to_string(self.prefix), self.p)

    def __lshift__(self, p):
        self.p = lae(self.p, p)

    def str(self):
        "wheeeee"
        return prefix_to_string(self.prefix).replace(" ", "")

class BeamSearch:
    def __init__(self, nbeams):
        self.nbeams = nbeams
        self.curbeams = {(0,): beam((0,), 0)}

    def add_logit(self, p):
        if len(self.curbeams) > self.nbeams:
            self.prune()
        self.oldbeams = self.curbeams
        self.curbeams = {}
        for beam in self.oldbeams.values():
            P = p + beam.p
            prefix = beam.prefix
            if prefix[-1] == 0:
                "The prefix ends with a blank symbol."
                "If we see a new blank character, then the prefix stays the same."
                "Otherwise, we extend the prefix by the new character."
                self.beam(prefix) << P[0]
                for i in range(1, nchars):
                    self.beam(prefix[:-1] + (i,)) << P[i]
            else:
                "The prefix ends with a nonblank symbol."
                "If we see a new blank character, then we extend the prefix by 0."
                "If we see the last character, then the prefix stays the same."
                "If we see any other nonblank character, then we extend the prefix by it."
                self.beam(prefix + (0,)) << P[0]
                for i in range(1, nchars):
                    if i == prefix[-1]:
                        self.beam(prefix) << P[i]
                    else:
                        self.beam(prefix + (i,)) << P[i]
        self.oldbeams = None

    def result(self):
        def combine_entries_with_spaces():
            for beam in [z for z in self.curbeams.values() if z.prefix[-1] == 0]:
                self.beam(beam.prefix[:-1]) << beam.p
                del self.curbeams[beam.prefix]
        combine_entries_with_spaces()
        return self.topbeams()

    def topbeams(self, nbeams=None):
        beamlist = list(self.curbeams.keys())
        beamlist.sort(key = lambda beam: -self.curbeams[beam].p)
        return [self.curbeams[beam] for beam in beamlist[:nbeams or self.nbeams]]

    def prune(self):
        self.curbeams = {beam.prefix: beam for beam in self.topbeams()}
        #whee
        #beamlist = list(self.curbeams.keys())
        #beamlist.sort(key = lambda beam: -self.curbeams[beam].p)
        #self.curbeams = {beam: self.curbeams[beam] for beam in beamlist[:self.nbeams]}

    def beam(self, prefix):
        if prefix not in self.curbeams:
            self.curbeams[prefix] = beam(prefix, -np.inf)
        return self.curbeams[prefix]

def beamsearcher(logits, nbeams):
    """Do a beam search on the logits.
    Logits should be shape (length, nchars)"""
    if len(logits.shape) == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    beas = BeamSearch(nbeams)
    for j in range(logits.shape[0]):
        beas.add_logit(logits[j, :])
    return beas.result()
