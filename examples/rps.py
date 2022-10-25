# Rock, paper, scissors example using pyactup

import pyactup
import random

DEFAULT_ROUNDS = 100
MOVES = ["paper", "rock", "scissors"]
N_MOVES = len(MOVES)

m = pyactup.Memory(noise=0.1)

def defeat_expectation(**kwargs):
    # Generate expectation matching supplied conditions and play the move that defeats.
    # If no expectation can be generate, chooses a move randomly.
    expectation = (m.retrieve(kwargs) or {}).get("move")
    if expectation:
        return MOVES[(MOVES.index(expectation) - 1) % N_MOVES]
    else:
        return random.choice(MOVES)

def safe_element(list, i):
    try:
        return list[i]
    except IndexError:
        return None

def main(rounds=DEFAULT_ROUNDS):
    # Plays multiple rounds of r/p/s of a lag 1 player (player1) versus a
    # lag 2 player (player2).
    plays1 = []
    plays2 = []
    score = 0
    for r in range(rounds):
        move1 = defeat_expectation(player="player2",
                                   ultimate=safe_element(plays2, -1))
        move2 = defeat_expectation(player="player1",
                                   ultimate=safe_element(plays1, -1),
                                   penultimate=safe_element(plays1, -2))
        winner = (MOVES.index(move2) - MOVES.index(move1) + N_MOVES) % N_MOVES
        score += -1 if winner == 2 else winner
        print("Round {:3d}\tPlayer 1: {:8s}\tPlayer 2: {:8s}\tWinner: {}\tScore: {:4d}".format(
            r, move1, move2, winner, score))
        m.learn({"player": "player1",
                 "ultimate": safe_element(plays1, -1),
                 "penultimate": safe_element(plays1, -2),
                 "move": move1})
        m.learn({"player": "player2",
                 "ultimate": safe_element(plays2, -1),
                 "move": move2},
                advance=2)
        plays1.append(move1)
        plays2.append(move2)


if __name__ == '__main__':
    main()
