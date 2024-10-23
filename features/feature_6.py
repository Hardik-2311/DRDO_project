from numpy import zeros
import numpy as np
from scipy.special import gammaincc as gammaincc
class Serial:

    @staticmethod
    def serial(binary_data: str, verbose=False, pattern_length=16):
        """Performs the Serial test for randomness.

        :param binary_data: The binary sequence to test.
        :param verbose: Whether to print detailed debug information.
        :param pattern_length: The length of the patterns to check.
        :return: A tuple containing the p-value and a boolean indicating if the test passed.
        """
        length_of_binary_data = len(binary_data)
        binary_data += binary_data[:(pattern_length - 1):]

        max_pattern = (1 << pattern_length) - 1

        # Step 02: Initialize the frequency counts for patterns of length m, m-1, m-2
        vobs_m = np.zeros(max_pattern + 1)
        vobs_m_minus_1 = np.zeros(1 << (pattern_length - 1))
        vobs_m_minus_2 = np.zeros(1 << (pattern_length - 2))

        # Count the occurrences of each pattern
        for i in range(length_of_binary_data):
            vobs_m[int(binary_data[i:i + pattern_length], 2)] += 1
            vobs_m_minus_1[int(binary_data[i:i + pattern_length - 1], 2)] += 1
            vobs_m_minus_2[int(binary_data[i:i + pattern_length - 2], 2)] += 1

        # Calculate Psi_m, Psi_m-1, Psi_m-2
        def calc_psi(vobs, pattern_len):
            return np.sum(vobs ** 2) * (1 << pattern_len) / length_of_binary_data - length_of_binary_data

        psi_m = calc_psi(vobs_m, pattern_length)
        psi_m_minus_1 = calc_psi(vobs_m_minus_1, pattern_length - 1)
        psi_m_minus_2 = calc_psi(vobs_m_minus_2, pattern_length - 2)

        # Calculate deltas
        delta_1 = psi_m - psi_m_minus_1
        delta_2 = psi_m - 2 * psi_m_minus_1 + psi_m_minus_2

        # Calculate the final p-value (using delta_1 as the main metric)
        p_value = gammaincc((1 << (pattern_length - 1)) / 2, delta_1 / 2.0)

        # Determine pass/fail based on p-value
        passed = p_value >= 0.01

        if verbose:
            print('Serial Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tPsi_m:', psi_m)
            print('\tPsi_m_minus_1:', psi_m_minus_1)
            print('\tPsi_m_minus_2:', psi_m_minus_2)
            print('\tDelta_1:', delta_1)
            print('\tDelta_2:', delta_2)
            print('\tP-Value:', p_value)
            print('DEBUG END.')

        return p_value, passed