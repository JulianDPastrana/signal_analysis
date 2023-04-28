import numpy as np
from scipy.fft import irfft
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class PowerSystem:
    def __init__(self, state: int = None) -> None:
        """
        Initialize the Power System class.
        Args:
        - state: seed value for numpy random state
        Initializes the PowerSystem class with the given numpy random seed value
        and sets the NETWORK_FREQUENCY, SAMPLING_FREQUENCY, and NUM_SAMPLES
        attributes.
        """
        self.NETWORK_FREQUENCY = 60
        self.SAMPLING_FREQUENCY = 1 / (100 * self.NETWORK_FREQUENCY)
        self.NUM_SAMPLES = 500
        self.rs = np.random.RandomState(state)

        print('\n'.join([
            f'Network Frequency: {self.NETWORK_FREQUENCY} [Hz]',
            f'Sampling Frequency: {1 / self.SAMPLING_FREQUENCY} [Samples / Seg]',
            f'Number of Samples: {self.NUM_SAMPLES}']))

    def __get_voltage_and_current(
            self,
            n_harmonics: int = 3,
            max_range: int = 12,
            v_max: float = 220 * np.sqrt(2),
            phi: float = 0,
            indexes: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the voltage and current based on the given parameters.
        Args:
        - n_harmonics: number of harmonics
        - max_range: maximum range for indexes
        - v_max: maximum voltage
        - phi: phase
        - indexes: list of indexes
        Returns:
        - voltage: voltage data
        - current: current data
        Given the specified number of harmonics, maximum range, maximum voltage,
        phase, and a list of indexes, this method returns voltage and current data
        based on the given parameters.
        """
        Vs_coeffs = np.zeros(self.NUM_SAMPLES // 2 + 1, dtype=complex)
        I_coeffs = np.zeros_like(Vs_coeffs)
        V_coeffs = np.zeros_like(Vs_coeffs)
        factor = int(self.NUM_SAMPLES * self.NETWORK_FREQUENCY * self.SAMPLING_FREQUENCY)

        R_line = self.rs.uniform(high=0.1)
        L_line = self.rs.uniform(high=1)

        R_load = self.rs.uniform(high=1)
        L_load, C_load = self.rs.uniform(low=-1, high=1, size=2)

        if indexes is None:
            indexes = self.rs.randint(2, max_range, size=n_harmonics)
            indexes = np.hstack([[1], indexes])

        for index in indexes:
            if index == 1:
                Vs_coeffs_mag = v_max * self.NUM_SAMPLES / 2
            else:
                Vs_coeffs_mag = self.rs.uniform(low=.5) * v_max * self.NUM_SAMPLES / (2 * index)

            k = index * factor
            Vs_coeffs_pha = self.rs.uniform(low=-1, high=1) * np.pi

            Vs_coeffs[k] = Vs_coeffs_mag * np.exp(1j * (phi + Vs_coeffs_pha))
            Z_line = self.__get_impedance(R_line, L_line, np.inf, index)
            Z_load = self.__get_impedance(R_load, L_load, C_load, index)

            I_coeffs[k] = Vs_coeffs[k] / (Z_line + Z_load)
            V_coeffs[k] = Z_load * I_coeffs[k]

        return irfft(V_coeffs), irfft(I_coeffs)

    def __get_impedance(self, R, L, C, k) -> complex:
        """
        Returns the impedance for a given set of parameters.
        Args:
        - R: resistance
        - L: inductance
        - C: capacitance
        - k: harmonic order
        Returns:
        - impedance: impedance value
        Given the specified resistance, inductance, capacitance, and harmonic order,
        this method calculates and returns the corresponding impedance value.
        """
        w = 2 * np.pi * self.NETWORK_FREQUENCY * k
        Z = R + 1j * w * L - 1j / (w * C)
        return Z

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns voltage and current data for each node in the power system.

        Args:
        - title: string, optional. The title to be used in the output. Default is "".

        Returns:
        - voltage_data: numpy array containing voltage data for each node
        - current_data: numpy array containing current data for each node

        This method returns two separate numpy arrays containing voltage and current
        data for each node in the power system, based on the output of the
        __get_voltage_and_current method. It also allows for an optional title to be specified.
        """
        voltage_data = np.empty((3, 3, self.NUM_SAMPLES))
        current_data = np.empty_like(voltage_data)

        for node in range(3):
            for ph in range(3):
                voltage, current = self.__get_voltage_and_current(phi=2 * np.pi / 3 * ph)
                voltage_data[node, ph] = voltage
                current_data[node, ph] = current

        return voltage_data, current_data


if __name__ == '__main__':
    system = PowerSystem()
    data = system.get_data()
    voltage, current = data
    fig, ax = plt.subplots(2, 1, sharex="all")
    ax[0].plot(voltage[0].T, label=['Phase A', 'Phase B', 'Phase C'])
    ax[0].set_ylabel("Voltage [V]")
    ax[0].legend()
    ax[1].plot(current[0].T)
    ax[1].set_ylabel("Current [A]")
    ax[1].set_xlabel("Samples")
    plt.show()
