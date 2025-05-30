import numpy as np
import math

class CustomFFT:
    @staticmethod
    def bit_reverse_permutation(x):
        N = len(x)
        j = 0
        for i in range(1, N):
            bit = N >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                x[i], x[j] = x[j], x[i]
        return x
    
    @staticmethod
    def fft(x):
        N = len(x)
        if N & (N - 1) != 0:
            raise ValueError(f"FFT size must be power of 2. Given: {N}")
        X = np.array(x, dtype=complex)
        X = CustomFFT.bit_reverse_permutation(X)
        levels = int(math.log2(N))
        for level in range(levels):
            m = 2 ** (level + 1)
            w_m = np.exp(-2j * np.pi / m)
            for i in range(0, N, m):
                w = 1.0
                for j in range(m // 2):
                    u_idx = i + j
                    v_idx = i + j + m // 2
                    u = X[u_idx]
                    v = X[v_idx]
                    X[u_idx] = u + w * v
                    X[v_idx] = u - w * v
                    w *= w_m
        return X
    
    @staticmethod
    def inverse_fft(X):
        N = len(X)
        X_conj = np.conj(X)
        x_conj = CustomFFT.fft(X_conj)
        return np.conj(x_conj) / N

