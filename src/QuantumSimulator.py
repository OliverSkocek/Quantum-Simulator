import numpy as np


class Basis:

    def __init__(self, number_fun, length, mass):
        self._number_fun = number_fun
        self._length = length
        self._mass = mass
        self._mask = np.ones(shape=(number_fun, number_fun), dtype=bool)

    def get_basis_representation(self, fun):
        pass

    def get_basis(self):
        pass

    def get_overlap_matrix(self):
        pass

    def get_kinetic_energy(self):
        pass

    def get_potential_energy(self, V, M=1000):
        """
        Computes potential Energy

        :param V: potential energy lambda.
        :param M: number of Equidistant point for numeric integration.
        """
        W = np.zeros((self._number_fun, self._number_fun))
        X = np.linspace(0, self._length, M)
        test = np.vectorize(V)(X[:-1])
        test = X[:-1][test != 0]
        boundary = (np.min(test), np.max(test))
        X = np.linspace(boundary[0], boundary[1], M)
        V_vec = np.vectorize(V)(X[:-1])
        for i in range(self._number_fun):
            indices = np.where(self._mask[i, :])[0]
            indices = indices[indices >= i]
            for j in indices:
                temp = np.sum(np.vectorize(self.get_basis(i))(X[:-1]) *
                              np.vectorize(self.get_basis(j))(X[:-1]) * V_vec * np.diff(X))
                W[i, j] = temp
                W[j, i] = temp
        return W

    def get_callable_from_base_representation(self, u):
        """
        compute the function phi representation of the wave.

        :param u: psi represented in the basis (psi_i)i=1:N.
        """
        return lambda x: np.sum([u[i] * self.get_basis(i)(x) for i in range(self._number_fun)])

    def get_one_parameter_unitary_group(self, potential):
        H = self.get_kinetic_energy() + self.get_potential_energy(potential)
        S = np.linalg.inv(self.get_overlap_matrix())
        H = np.matmul(S, H)
        D, M = np.linalg.eig(H)
        return lambda t: np.matmul(M, np.matmul(np.diag(np.exp(1j * D * t)), M.T))


class ShapeFunctions(Basis):
    def __init__(self, number_fun, length, mass):
        super().__init__(number_fun, length, mass)
        temp = np.roll(np.eye(number_fun), 1)
        temp[-1, 0] = False
        self._mask = np.eye(number_fun) + temp + temp.T
        self.X = np.linspace(0, self._length, self._number_fun + 2)

    def get_basis_representation(self, fun):
        return fun(self.X[1:-1])

    def get_basis(self):
        return lambda k: lambda x: (
            (x / (self.X[k + 1] - self.X[k]) - self.X[k] / (self.X[k + 1] - self.X[k]))
            if x < self.X[k + 1] else (-x / (self.X[k + 2] - self.X[k + 1]) + self.X[k + 2] / (self.X[k + 2] - self.X[k + 1]))) \
            if self.X[k] < x < self.X[k + 2] else 0 * x

    def get_overlap_matrix(self):
        a = np.diff(self.X)
        diag = np.eye(self.X.size - 2) * (a[:-1] + a[1:]) / 3
        n_diag = np.roll(np.eye(self.X.size - 2), shift=1, axis=1)
        n_diag *= a[:-1] / 6
        n_diag[-1, 0] = 0.0
        return diag + n_diag + n_diag.T

    def get_kinetic_energy(self):
        a = np.diff(self.X)
        diag = np.eye(self.X.size - 2) * (1 / a[:-1] + 1 / a[1:])
        n_diag = np.roll(np.eye(self.X.size - 2), shift=1, axis=1)
        n_diag *= -1 / a[:-1]
        n_diag[-1, 0] = 0.0
        return (1 / (2 * self._mass)) * (diag + n_diag + n_diag.T)




def sine_basis(length, number_fun):
    return lambda n: lambda x: np.sqrt(2 / length) * np.sin(np.pi * (n + 1) * x / length)

def init_psi(N, phi, M=1000):
    """
    computes the basis representation of phi.
    :param N: number of basis vectors.
    :param M: number of äquidistant subdivisions for numeric integration.
    :param phi: lambda of the wave function.
    """
    X = np.linspace(0, L, M)
    ls = list()
    for i in range(N):
        ls.append(np.sum(np.vectorize(psi(i))(X[:-1])*np.vectorize(phi)(X[:-1])*np.diff(X)))
    return np.array(ls)





def assemble_hamiltonian(X, V, M=1000):
    return get_kinetic_energy(X) + compute_potential_energy(X, V, M)


def get_integrator(t, X, V, M=10000):
    H = assemble_hamiltonian(X, V)
    # D, U = np.linalg.eig(get_overlap_integral(X))
    # S = np.matmul(np.matmul(U, np.diag(1/np.sqrt(D))), U.T)
    D, U = np.linalg.eig(np.matmul(np.linalg.inv(get_overlap_integral(X)), H))

    # rewrite this with a solver and two 1-d arrays instead of full matrix.
    return np.matmul(np.matmul(U, np.diag(np.exp(D * t * 1j))), U.T)
