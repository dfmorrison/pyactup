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
    for c, o in (("safe", 1), ("risky", 0), ("risky", 2)):
        m.learn(choice=c, outcome=o, advance=0)
    m.advance()
    for r in range(ROUNDS):
        choice, bv = m.best_blend("outcome", ("safe", "risky"), "choice")
        if choice == "risky":
            payoff = 3 if random.random() < 1/3 else 0
            risky_chosen[r] += 1
        else:
            payoff = 1
        m.learn(choice=choice, outcome=payoff)

plt.plot(range(ROUNDS), [ v / PARTICIPANTS for v in risky_chosen])
plt.ylim([0, 1])
plt.ylabel("fraction choosing risky")
plt.xlabel("round")
plt.title(f"Safe (1 always) versus risky (3 × ⅓, 0 × ⅔)\nσ={m.noise}, d={m.decay}")
plt.show()
