import numpy as np
from scipy.special import gammaincc

class ComplexityTest:

    @staticmethod
    def linear_complexity_test(binary_data: str, verbose=False, block_size=500):
        """
        Optimized version of the linear complexity test.
        """

        length_of_binary_data = len(binary_data)
        degree_of_freedom = 6

        #  π0 = 0.010417, π1 = 0.03125, π2 = 0.125, π3 = 0.5, π4 = 0.25, π5 = 0.0625, π6 = 0.020833
        pi = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
        mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2

        number_of_block = length_of_binary_data // block_size

        if number_of_block > 1:
            blocks = [binary_data[i * block_size:(i + 1) * block_size] for i in range(number_of_block)]

            complexities = [ComplexityTest.berlekamp_massey_algorithm(block) for block in blocks]

            t = np.array([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])

            # Histogram (more efficient way to handle bins)
            bins = np.array([-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf])
            vg, _ = np.histogram(t, bins=bins)

            # Chi-square test statistic (improved way to compute xObs)
            xObs = np.sum(((vg - number_of_block * np.array(pi)) ** 2) / (number_of_block * np.array(pi)))

            # P-Value using gammaincc
            p_value = gammaincc(degree_of_freedom / 2.0, xObs / 2.0)

            if verbose:
                print('Linear Complexity Test DEBUG BEGIN:')
                print(f'\tLength of input:\t{length_of_binary_data}')
                print(f'\tLength in bits of a block:\t{block_size}')
                print(f'\tDegree of Freedom:\t\t{degree_of_freedom}')
                print(f'\tNumber of Blocks:\t{number_of_block}')
                print(f'\tValue of Vs:\t\t{vg}')
                print(f'\txObs:\t\t\t\t{xObs}')
                print(f'\tP-Value:\t\t\t{p_value}')
                print('DEBUG END.')

            return p_value, (p_value >= 0.01)
        else:
            return -1.0, False

    @staticmethod
    def berlekamp_massey_algorithm(block_data):
        """
        Optimized implementation of the Berlekamp Massey Algorithm.
        """
        n = len(block_data)
        block_array = np.array([int(bit) for bit in block_data])

        c = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        c[0], b[0] = 1, 1
        l, m, i = 0, -1, 0

        while i < n:
            if l > 0:
                v = block_array[i - l:i][::-1]
                cc = c[1:l + 1]
                d = (block_array[i] + np.dot(v, cc)) % 2
            else:
                d = block_array[i] % 2

            if d == 1:
                temp_c = c.copy()
                p = np.zeros(n, dtype=int)

                for j in range(l):
                    if b[j] == 1:
                        p[j + i - m] = 1

                c = (c + p) % 2

                if l <= 0.5 * i:
                    l = i + 1 - l
                    m = i
                    b = temp_c
            i += 1

        return l
