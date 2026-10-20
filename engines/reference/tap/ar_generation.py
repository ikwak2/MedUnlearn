"""CPU acceleration of the official causal AR recurrence, preserving float32 operations."""
import numpy as np
from numba import njit


@njit(cache=True)
def recurrence(signals, coefficients):
    for n in range(signals.shape[0]):
        for c in range(3):
            for i in range(signals.shape[2]-2):
                for j in range(signals.shape[3]-2):
                    value = np.float32(0)
                    for u in range(3):
                        for v in range(3):
                            value += signals[n,c,i+u,j+v]*coefficients[n,c,u,v]
                    signals[n,c,i+2,j+2] = value
    return signals
