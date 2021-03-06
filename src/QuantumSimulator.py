import numpy as np


class Basis:

    def __init__(self, number_fun, length, mass, integration_num=1000):
        self._number_fun = number_fun
        self._length = length
        self._mass = mass
        self._mask = np.ones(shape=(number_fun, number_fun), dtype=bool)
        self._integration_num = integration_num

    def get_basis_representation(self, fun):
        pass

    def get_basis(self):
        pass

    def get_overlap_matrix(self):
        pass

    def get_kinetic_energy(self):
        pass

    def get_potential_energy(self, V):
        """
        Computes potential Energy

        :param V: potential energy lambda.
        :param M: number of Equidistant point for numeric integration.
        """
        W = np.zeros((self._number_fun, self._number_fun), dtype=complex)
        X = np.linspace(0, self._length, self._integration_num)
        test = np.vectorize(V)(X[:-1])
        test = X[:-1][test != 0]
        boundary = (np.min(test), np.max(test))
        X = np.linspace(boundary[0], boundary[1], self._integration_num)
        V_vec = np.vectorize(V)(X[:-1])
        for i in range(self._number_fun):
            indices = np.where(self._mask[i, :])[0]
            indices = indices[indices >= i]
            for j in indices:
                temp = np.sum(np.conj(np.vectorize(self.get_basis()(i))(X[:-1])) *
                              np.vectorize(self.get_basis()(j))(X[:-1]) * V_vec * np.diff(X))
                W[i, j] = temp
                W[j, i] = np.conj(temp)
        return W

    def get_callable_from_base_representation(self, u):
        """
        compute the function phi representation of the wave.

        :param u: psi represented in the basis (psi_i)i=1:N.
        """
        return lambda x: np.sum([u[i] * self.get_basis()(i)(x) for i in range(self._number_fun)])

    def get_one_parameter_unitary_group(self, potential):
        # ist basiswechsel zu inverse square root von S numerisch stabiler?
        H = self.get_kinetic_energy() + self.get_potential_energy(potential)
        S = np.linalg.inv(self.get_overlap_matrix())
        H = np.matmul(S, H)
        D, M = np.linalg.eig(H)
        return lambda t: np.matmul(M, np.matmul(np.diag(np.exp(1j * D * t)), np.matrix.getH(M)))

    def get_density_from_base_representation(self, u):
        return lambda x: np.square(np.abs(self.get_callable_from_base_representation(u)(x)))


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
            if x < self.X[k + 1] else (
                    -x / (self.X[k + 2] - self.X[k + 1]) + self.X[k + 2] / (self.X[k + 2] - self.X[k + 1]))) \
            if self.X[k] <= x < self.X[k + 2] else 0 * x

    def get_overlap_matrix(self):
        a = np.diff(self.X)
        diag = np.eye(self._number_fun) * (a[:-1] + a[1:]) / 3
        n_diag = np.roll(np.eye(self._number_fun), shift=1, axis=1)
        n_diag *= a[:-1] / 6
        n_diag[-1, 0] = 0.0
        return diag + n_diag + n_diag.T

    def get_kinetic_energy(self):
        a = np.diff(self.X)
        diag = np.eye(self._number_fun) * (1 / a[:-1] + 1 / a[1:])
        n_diag = np.roll(np.eye(self._number_fun), shift=1, axis=1)
        n_diag *= -1 / a[:-1]
        n_diag[-1, 0] = 0.0
        return (1 / (2 * self._mass)) * (diag + n_diag + n_diag.T)


class Sine(Basis):
    def __init__(self, number_fun, length, mass):
        super().__init__(number_fun, length, mass)

    def get_basis_representation(self, fun):
        """
        computes the basis representation of fun.

        :param fun: lambda of the wave function.
        """
        X = np.linspace(0, self._length, self._integration_num)
        ls = list()
        for i in range(self._number_fun):
            ls.append(np.sum(np.vectorize(self.get_basis()(i))(X[:-1]) * np.vectorize(fun)(X[:-1]) * np.diff(X)))
        return np.array(ls)

    def get_basis(self):
        return lambda n: lambda x: np.sqrt(2 / self._length) * np.sin(np.pi * (n + 1) * x / self._length)

    def get_overlap_matrix(self):
        return np.eye(self._number_fun)

    def get_kinetic_energy(self):
        return (1 / (2 * self._mass)) * np.square(np.pi / self._length) * np.diag(
            np.square(np.arange(self._number_fun) + 1))


class Fourier(Sine):
    def __init__(self, number_fun, length, mass):
        super().__init__(2 * (number_fun // 2) + 1, length, mass)

    def get_basis_representation(self, fun):
        """
        computes the basis representation of fun.

        :param fun: lambda of the wave function.
        """
        X = np.linspace(0, self._length, self._integration_num)
        ls = list()
        for i in range(self._number_fun):
            ls.append(
                np.sum(np.conj(np.vectorize(self.get_basis()(i))(X[:-1])) * np.vectorize(fun)(X[:-1]) * np.diff(X)))
        return np.array(ls)

    def get_basis(self):
        return lambda n: lambda x: np.sqrt(1 / self._length) * np.exp(
            2 * np.pi * 1j * (n - self._number_fun // 2) * x / self._length)

    def get_overlap_matrix(self):
        return self.get_potential_energy(V=lambda x: 1.0 + (x * 0.0))

    def get_kinetic_energy(self):
        return (1 / (2 * self._mass)) * np.square(2 * np.pi / self._length) * np.diag(
            np.square(np.arange(-self._number_fun // 2, self._number_fun // 2)))


class ShapeFunctionsRing(ShapeFunctions):
    def __init__(self, number_fun, length, mass):
        super().__init__(number_fun, length, mass)
        temp = np.roll(np.eye(number_fun), 1)
        self._mask = np.eye(number_fun) + temp + temp.T
        self.X = np.linspace(0, self._length, self._number_fun + 1)
        self.faces = [(self.X[(j - 1) % self._number_fun], self.X[j], self.X[(j + 1) % self._number_fun])
                      for j in range(self._number_fun)]

    def get_basis_representation(self, fun):
        return fun(self.X[:-1])

    def _get_face_base_fn(self, face, x):
        face = list(face)
        if face[0] < face[1]:
            if face[2] < face[1]:
                face[2] = self._length
            if face[0] < x <= face[1]:
                return x / (face[1] - face[0]) - face[0] / (face[1] - face[0])
            elif face[1] < x <= face[2]:
                return - x / (face[2] - face[1]) + face[2] / (face[2] - face[1])
            else:
                return 0 * x
        else:
            if face[1] <= x <= face[2]:
                return - x / (face[2] - face[1]) + face[2] / (face[2] - face[1])
            elif face[0] <= x <= self._length:
                return x / (self._length - face[0]) - face[0] / (self._length - face[0])
            else:
                return 0 * x

    def get_basis(self):
        return lambda k: lambda x: self._get_face_base_fn(self.faces[k], x)

    def get_overlap_matrix(self):
        return self.get_potential_energy(V=lambda x: 1.0 + (x * 0.0))

    def get_kinetic_energy(self):
        a = np.diff(self.X)
        diag = np.eye(self._number_fun) * (1 / a + 1 / np.roll(a, shift=1))
        n_diag = np.diag(-1 / a)
        n_diag = np.roll(n_diag, shift=1, axis=1)
        return (1 / (2 * self._mass)) * (diag + n_diag + n_diag.T)
