 # -*- coding:<some encoding> -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import irfft

def random_data_generator(state=None):

    rs = np.random.RandomState()

    if state:
        rs.set_state(state)
    else:
        state = rs.get_state()



    A = rs.uniform(0.1, 1)*220*np.sqrt(2)
    phi = rs.uniform(-1, 1)*np.pi
    phi0 = np.array([0, -120, 120])*np.pi/180

    F = 60
    Fs = 6000
    N = Fs//F * 5
    t = np.linspace(0, 5/F, N)
    voltages = np.array(
        [A*np.sin(2*np.pi*F*t + phi0[0] + phi),
        A*np.sin(2*np.pi*F*t + phi0[1] + phi),
        A*np.sin(2*np.pi*F*t + phi0[2] + phi)]
    )

    currents = np.empty(shape=(3, N))
    for i in range(3):
        c_ = np.zeros(N//2 + 1, dtype=complex)
        for k in range(1, 10, 2):
            index = 60*k*N//Fs
            cmag = rs.uniform(low=0.5) * 220*np.sqrt(2) * N / (2*k**1.5)
            cphase = rs.uniform(-.1, .1)*np.pi
            c_[index] = cmag * np.exp(1j*(phi0[i]+cphase))

        currents[i] = irfft(c_)

    data_info = {
        'voltages': voltages,
        'currents': currents,
        'F': F,
        'Fs': Fs,
        'state': state
    }

    return data_info
