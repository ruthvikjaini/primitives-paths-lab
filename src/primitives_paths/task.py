"""Synthetic composition task from Qu (2026), arXiv:2611.11260, Sec. 5.

Five-digit strings; two primitive operations:
  F: rotate the string left by one position.
  G: increment the first digit by three modulo ten.
A problem is a sequence of 2-4 operations containing both F and G. A
programmatic verifier checks every intermediate state and the final answer.

Token ids: digits 0-9 -> 0-9, F -> 10, G -> 11, '=' -> 12, EOS -> 13, PAD -> 14.
"""

import numpy as np

DIGITS = "0123456789"
F_ID = 10
G_ID = 11
SEP_ID = 12
EOS_ID = 13
PAD_ID = 14
VOCAB_SIZE = 15
MAX_LEN = 40

OP_IDS = {"F": F_ID, "G": G_ID}
ID_OPS = {F_ID: "F", G_ID: "G"}


def apply_op(state, op):
    """Apply one primitive to a 5-digit string."""
    if op == "F":
        return state[1:] + state[0]
    if op == "G":
        return str((int(state[0]) + 3) % 10) + state[1:]
    raise ValueError(f"unknown op {op!r}")


def apply_ops(state, ops):
    """Return the list of intermediate states after each op in `ops`."""
    states = []
    cur = state
    for op in ops:
        cur = apply_op(cur, op)
        states.append(cur)
    return states


def random_string(rng):
    return "".join(rng.choice(list(DIGITS)) for _ in range(5))


def make_problem(rng, m=None):
    """Sample a composition problem: (start_string, ops list).

    m is the number of ops (2-4 by default); the sequence always contains
    both F and G, as in the paper.
    """
    if m is None:
        m = int(rng.integers(2, 5))
    while True:
        ops = [str(x) for x in rng.choice(["F", "G"], size=m)]
        if "F" in ops and "G" in ops:
            break
    return random_string(rng), ops


def prompt_tokens(start, ops):
    """Prompt: 5 digits, op ids, '='."""
    toks = [int(ch) for ch in start]
    toks += [OP_IDS[op] for op in ops]
    toks.append(SEP_ID)
    return toks


def response_tokens(states):
    """Response: states joined by '=', terminated by EOS."""
    toks = []
    for i, s in enumerate(states):
        if i > 0:
            toks.append(SEP_ID)
        toks += [int(ch) for ch in s]
    toks.append(EOS_ID)
    return toks


def full_sequence(start, ops):
    """Prompt + response token ids for one problem."""
    states = apply_ops(start, ops)
    return prompt_tokens(start, ops) + response_tokens(states)


def parse_response(tokens):
    """Parse response tokens into a list of 5-digit state strings.

    Returns None if the tokens do not parse as states joined by '=' and
    terminated by EOS.
    """
    states = []
    cur = []
    for t in tokens:
        if t == EOS_ID:
            if len(cur) != 5:
                return None
            states.append("".join(str(d) for d in cur))
            return states
        if t == SEP_ID:
            if len(cur) != 5:
                return None
            states.append("".join(str(d) for d in cur))
            cur = []
        elif 0 <= t <= 9:
            cur.append(t)
            if len(cur) > 5:
                return None
        else:
            return None
    return None  # no EOS


def verify_response(start, ops, tokens):
    """True iff `tokens` are exactly the correct intermediate states + EOS."""
    states = parse_response(tokens)
    if states is None:
        return False
    return states == apply_ops(start, ops)


def single_op_prompt(start, op):
    """Isolated primitive probe: one op only, e.g. '12345 F ='."""
    return prompt_tokens(start, [op])


def single_op_response(start, op):
    return response_tokens([apply_op(start, op)])


def handoff_prompt(start, ops, t):
    """Prompt with the first `t` reference states supplied (paper Sec. 4).

    t=0 is the autonomous prompt. For t>=1 the supplied states are appended
    after the '=', each followed by '=' so the model must continue with the
    next state.
    """
    toks = prompt_tokens(start, ops)
    if t == 0:
        return toks
    states = apply_ops(start, ops)
    for s in states[:t]:
        toks += [int(ch) for ch in s]
        toks.append(SEP_ID)
    return toks


def handoff_response(start, ops, t):
    """The completion the model must produce given a t-step handoff."""
    states = apply_ops(start, ops)
    return response_tokens(states[t:])
