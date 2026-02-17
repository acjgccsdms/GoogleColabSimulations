"""
random_walk.py
--------------

Simple random walk simulations for Monte Carlo experiments.

Includes:
- 1D walk
- 2D walk
- many trials
- mean squared displacement (diffusion metric)

"""

import numpy as np

# The two walks
def random_walk_1d(n_steps: int, p: float = .5)-> np.ndarray:
    """
    Simulate a 1D random walk.

    Params
    ------
    n_steps : number of time steps
    p: probability of a +1 step

    Returns
    -------
    positions : cumulative position array (length n_steps)
    """
    steps = np.where(np.random.rand(n_steps) < p, 1, -1)
    return np.cumsum(steps)

def random_walk_2d(n_steps: int) -> np.ndarray:
    """
    2D lattice random walk. 

    Returns
    -------
    positions : (n_steps, 2) array of x,y coords.
    """

    directions = np.array([[1,0],[-1,0],[0,1],[0,-1]])

    choices = directions[np.random.randint(0, 4, size=n_steps)]
    return np.cumsum(choices, axis = 0)

def many_walks_1D(n_steps: int, n_trials: int) -> np.ndarray:
    """
    Run multiple 1D walks at once. 

    Returns
    -------
    shape (n_trials, n_steps)
    """
    steps = np.random.choice([-1,1], size = (n_trials, n_steps))
    return np.cumsum(steps, axis = 1)

def mean_squared_displacement(walks: np.ndarray) -> np.ndarray:
    """
    Compute E[X^2] across many trials.

    walks: shape (n_trials, n_steps)

    Returns
    -------
    MSD array of length n_steps
    """
    return np.mean(walks**2, axis = 0)

def theoretical_msd(n_steps: int)-> np.ndarray:
    """
    For a simple symettric walk E[X^2] = t
    """
    return np.arange(1, n_steps+1)

def random_walk_absorbing(left: int, 
    right: int,
    start: int = 0,
    p: float = .5,
    max_steps: int = 10000):

    """
    Run a walk until it hits a boundary.

    Returns
    -------
    path: full path until absorption
    hit_time: first hitting time
    hit_location: which bounday is hit
    """

    pos = start
    path = [pos]
    for t in range(1, max_steps +1):
        step = 1 if np.random.rand() < p else -1

        pos += step
        path.append(pos)

        if pos <= left or pos>= right: 
            return np.array(path), t, pos
    return np.array(path), max_steps, pos

def hitting_time(
    left: int,
    right: int,
    n_trials: int = 1000,
    start: int = 0
    ):
    """
    Estimate the hitting time for many random walks.

    Returns
    -------
    times: array of hitting times
    hit_probs: prob of hitting each boundary
    """
    times = []
    hits = []

    for path in range(n_trials):
        path, t, loc = random_walk_absorbing(left, right, start)
        times.append(t)
        hits.append(loc)

    times = np.array(times)
    hits = np.array(hits)

    prob_left = np.mean(hits == left)
    prob_right = np.mean(hits == right)

    return times, (prob_left, prob_right)
    









