import pyactup
import random

import matplotlib.pyplot as plt

from tqdm import tqdm

PARTICIPANTS = 10_000
ROUNDS = 60

risky_chosen = [0] * ROUNDS
m = pyactup.Memory()
for p in tqdm(range(PARTICIPANTS)):
    m.reset()
    # prepopulate some instances to ensure initial exploration
    m.learn(choice="safe", outcome=1)
    m.learn(choice="risky", outcome=0)
    m.learn(choice="risky", outcome=2)
    m.advance()
    for r in range(ROUNDS):
        safe_bv = m.blend("outcome", choice="safe")
        risky_bv = m.blend("outcome", choice="risky")
        if risky_bv > safe_bv or (risky_bv == safe_bv and random.random() < 0.5):
            choice = "risky"
            payoff = 3 if random.random() < 1/3 else 0
            risky_chosen[r] += 1
        else:
            choice = "safe"
            payoff = 1
        m.learn(choice=choice, outcome=payoff)
        m.advance()

plt.plot(range(ROUNDS), [ v / PARTICIPANTS for v in risky_chosen])
plt.ylim([0, 1])
plt.ylabel("fraction choosing risky")
plt.xlabel("round")
plt.title(f"Safe (1 always) versus risky (3 × ⅓, 0 × ⅔)\nσ={m.noise}, d={m.decay}")
plt.show()

