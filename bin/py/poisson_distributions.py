import numpy as np

from drmdp import rewdelay


def main():
    for lam in (3, 10, 15):
        reward_delay = rewdelay.ClippedPoissonDelay(lam=lam)
        values = np.array([reward_delay.sample() for _ in range(1_000_000)])
        print("Delay:", lam, "Min:", np.min(values), "Max:", np.max(values))


if __name__ == "__main__":
    main()
