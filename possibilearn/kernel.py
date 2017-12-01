r"""
Module handling kernel models in yaplf

Module :mod:`yaplf.models.kernel` contains all the classes handling kernel
models in yaplf.

AUTHORS:

- Dario Malchiodi (2010-02-15): initial version

"""

#*****************************************************************************
#       Copyright (C) 2010 Dario Malchiodi <malchiodi@di.unimi.it>
#
# This file is part of yaplf.
# yaplf is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
# yaplf is distributed in the hope that it will be useful, but without any
# warranty; without even the implied warranty of merchantability or fitness
# for a particular purpose. See the GNU Lesser General Public License for
# more details.
# You should have received a copy of the GNU Lesser General Public License
# along with yaplf; if not, see <http://www.gnu.org/licenses/>.
#
#*****************************************************************************


from numpy import dot, exp, array, shape, tanh
from numpy.linalg import norm


class Kernel(object):
    r"""
    Base class for kernels. Each subclass should implement the method
    :meth:`compute`, having as input two patterns and returning the kernel
    value. Subclasses essentially implements the strategy pattern [Gamma et
    al., 1995].
    
    The class defaults its :obj:`precomputed` field to ``False``. Subclassses
    for precomputed kernels should override this setting.

    EXAMPLES:

    See the examples section for concrete subclasses, such as
    :class:`GaussianKernel` in this package.

    REFERENCES:

    [Gamma et al., 1995] Erich Gamma, Richard Helm, Ralph Johnoson, John
    Vlissides, Design patterns: elements of reusable object-oriented software,
    Reading, Mass.: Addison-Wesley, 1995 (ISBN: 0201633612).

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self):
        r"""
        See :class:`Kernel` for full documentation.
        """

        self.precomputed = False

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value for a given pair of arguments.

        :param arg_1: first kernel argument.

        :param arg_2: second kernel argument.

        :returns: kernel value

        :rtype: float

        EXAMPLES:

        When invoked in the base class, this method raises a
        :exc:`NotImplementedError`.

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        raise NotImplementedError(
            'this class does not implement compute method')

    @classmethod
    def get_default(cls):
        r"""
        Factory method returning the default kernel to be used.

        :returns: default kernel class

        :rtype: Kernel

        EXAMPLES:

        >>> from yaplf.models.kernel import Kernel
        >>> Kernel.get_default()
        LinearKernel()

        AUTHORS:

        - Dario Malchiodi (2011-11-27)

        """

        return LinearKernel()

class LinearKernel(Kernel):
    r"""
    Linear kernel corresponding to dot product in the original space.

    EXAMPLES:

    Arguments of a dot product are numeric list or tuples having the same
    length, expressed as arguments of method :meth:`compute`:

    >>> from yaplf.models.kernel import LinearKernel
    >>> k = LinearKernel()
    >>> k.compute((1, 0, 2), (-1, 2, 5))
    9.0
    >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
    3.3200000000000003

    List and tuples can intertwine as arguments:

    >>> k.compute((1.2, -0.4, -2), [4, 1.2, .5])
    3.3200000000000003

    Specification of iterables having unequal length causes a :exc:`ValueError`
    to be thrown.

    >>> k.compute((1, 0, 2), (-1, 2))
    Traceback (most recent call last):
    ...
    ValueError: objects are not aligned

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def compute(self, arg_1, arg_2):
        r"""
        Compute the dot product between :obj:`arg_1` and obj:`arg_2`, where the
        dot product :math:`x \cdot y` is intended as the quantity
        :math:`\sum_{i=1}^n x_i y_i`, :math:`n` being the dimension of both
        :math:`x` and :math:`y`.

        :param arg_1: first dot product argument.

        :type arg_1: iterable

        :param arg_2: second dot product argument.

        :type arg_2: iterable

        :returns: kernel value.

        :rtype: float

        EXAMPLES:

        Arguments of a dot product are numeric list or tuples having the same
        length, expressed as arguments of the function :meth:`compute`:

        >>> from yaplf.models.kernel import LinearKernel
        >>> k = LinearKernel()
        >>> k.compute((1, 0, 2), (-1, 2, 5))
        9.0
        >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
        3.3200000000000003

        List and tuples can intertwine as arguments:

        >>> k.compute((1.2, -0.4, -2), [4, 1.2, .5])
        3.3200000000000003

        Specification of iterables having unequal length causes a
        :exc:`ValueError` to be thrown.

        >>> k.compute((1, 0, 2), (-1, 2))
        Traceback (most recent call last):
        ...
        ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return float(dot(arg_1, arg_2))

    def __repr__(self):
        return 'LinearKernel()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash('LinearKernel')

    def __nonzero__(self):
        return True


class PolynomialKernel(Kernel):
    r"""
    Polynomial kernel inducing in the original space polynomial surfaces.

    :param degree: polynomial degree.

    :type degree: integer

    EXAMPLES:

    A :class:`PolynomialKernel` object is obtained in function of its degree:



    >>> from yaplf.models.kernel import PolynomialKernel
    >>> k = PolynomialKernel(2)

    Only positive integers can be used as polynomial degree, for a
    :exc:`ValueError` is otherwise thrown:

    >>> PolynomialKernel(3.2)
    Traceback (most recent call last):
    ...
    ValueError: 3.2 is not usable as a polynomial degree
    >>> PolynomialKernel(-2)
    Traceback (most recent call last):
    ...
    ValueError: -2 is not usable as a polynomial degree

    Arguments of a polynomial kernel are numeric list or tuples (possibily
    intertwined) having the same length, expressed as arguments of the
    :meth:`compute` method:

    >>> k.compute((1, 0, 2), (-1, 2, 5))
    100.0
    >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
    18.662400000000002
    >>> k = PolynomialKernel(5)
    >>> k.compute((1, 0, 2), [-1, 2, 5])
    100000.0
    >>> k.compute((1.2, -0.4, -2), (4, 1.2, .5))
    1504.5919506432006

    Specification of iterables having unequal length causes a :exc:`ValueError`
    to be thrown.

    >>> k.compute((1, 0, 2), (-1, 2))
    Traceback (most recent call last):
    ...
    ValueError: objects are not aligned

    AUTHORS:

    - Dario Malchiodi (2010-02-22)

   """

    def __init__(self, degree):
        r"""
        See :class:`PolynomialKernel` for full documentation.

        """

        Kernel.__init__(self)
        if degree > 0 and int(degree) == degree:
            self.degree = degree
        else:
            raise ValueError(str(degree) +
                ' is not usable as a polynomial degree')

    def compute(self, arg_1, arg_2):
        r"""
        Compute the polynomial kernel between :obj:`arg_1` and :obj:`arg_2`,
        where the kernel value :math:`k(x_1, x_2)` is intended as the quantity
        :math:`(x_1 \cdot x_2 + 1)^d`, :math:`d` being the polynomial degree of
        the kernel.

        :param arg_1: first argument to the polynomial kernel.

        :param arg_2: second argument to the polynomial kernel.

        :returns: kernel value.

        :rtype: float

        EXAMPLES:

        Arguments of :meth:`compute` are numeric list or tuples (possibily
        intertwined) having the same length:

        >>> from yaplf.models.kernel import PolynomialKernel
        >>> k = PolynomialKernel(2)
        >>> k.compute((1, 0, 2), (-1, 2, 5))
        100.0
        >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
        18.662400000000002
        >>> k = PolynomialKernel(5)
        >>> k.compute((1, 0, 2), [-1, 2, 5])
        100000.0
        >>> k.compute((1.2, -0.4, -2), (4, 1.2, .5))
        1504.5919506432006

        Specification of iterables having unequal length causes a
        :exc:`ValueError` to be thrown.

        >>> k.compute((1, 0, 2), (-1, 2))
        Traceback (most recent call last):
        ...
        ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return float((dot(arg_1, arg_2) + 1) ** self.degree)

    def __repr__(self):
        return 'PolynomialKernel(' + repr(self.degree) + ')'

    def __str___(self):
        return self.__repr__()


class HomogeneousPolynomialKernel(Kernel):
    r"""
    Homogenous polynomial kernel inducing in the original space
    *homogeneous* polynomial surfaces.

    :param degree: polynomial degree.

    :type degree: integer

    EXAMPLES:

    An :class:`HomogeneousPolynomialKernel` object is obtained in function of
    its degree:

    >>> from yaplf.models.kernel import HomogeneousPolynomialKernel
    >>> k = HomogeneousPolynomialKernel(2)

    Only positive integers can be used as polynomial degree, as a
    :exc:`ValueError` is otherwise thrown:

    >>> HomogeneousPolynomialKernel(3.2)
    Traceback (most recent call last):
    ...
    ValueError: 3.2 is not usable as a polynomial degree
    >>> HomogeneousPolynomialKernel(-2)
    Traceback (most recent call last):
    ...
    ValueError: -2 is not usable as a polynomial degree

    Arguments of an homogeneous polynomial kernel are numeric list or tuples
    (possibily intertwined) having the same length, expressed as arguments of
    method :meth:`compute`:

    >>> k.compute((1, 0, 2), (-1, 2, 5))
    81.0
    >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
    11.022400000000001
    >>> k = HomogeneousPolynomialKernel(5)
    >>> k.compute((1, 0, 2), [-1, 2, 5])
    59049.0
    >>> k.compute((1.2, -0.4, -2), (4, 1.2, .5))
    403.35776184320019

    Specification of iterables having unequal length causes a :exc:`ValueError`
    to be thrown:

    >>> k.compute((1, 0, 2), (-1, 2))
    Traceback (most recent call last):
    ...
    ValueError: objects are not aligned

    AUTHORS:

    - Dario Malchiodi (2010-02-22)


    """

    def __init__(self, degree):
        r"""
        See :class:`HomogeneousPolynomialKernel` for full documentation.

        """

        Kernel.__init__(self)
        if degree > 0 and int(degree) == degree:
            self.degree = degree
        else:
            raise ValueError(str(degree) +
                ' is not usable as a polynomial degree')

    def compute(self, arg_1, arg_2):
        r"""
        Compute the homogeneous polynomial kernel between :obj:`arg_1` and
        :obj:`arg_2`, where the kernel value :math;`k(x_1, x_2)` is intended as
        the quantity :math:`(x_1 \cdot x_2)^d`, :math:`d` being the polynomial
        degree of the kernel.

        :param arg_1: first argument to the homogeneous polynomial kernel.

        :param arg_2: second argument to the homogeneous polynomial kernel.

        :returns: kernel value.

        :rtype: float

        EXAMPLES:

        Arguments of :meth:`compute` are numeric list or tuples (possibily
        intertwined) having the same length:

        >>> from yaplf.models.kernel import HomogeneousPolynomialKernel
        >>> k = HomogeneousPolynomialKernel(2)
        >>> k.compute((1, 0, 2), (-1, 2, 5))
        81.0
        >>> k.compute([1.2, -0.4, -2], [4, 1.2, .5])
        11.022400000000001
        >>> k = HomogeneousPolynomialKernel(5)
        >>> k.compute((1, 0, 2), [-1, 2, 5])
        100000.0
        >>> k.compute((1.2, -0.4, -2), (4, 1.2, .5))
        1504.5919506432006

        Specification of iterables having unequal length causes a
        :exc:`ValueError` to be thrown.

        >>> k.compute((1, 0, 2), (-1, 2))
        Traceback (most recent call last):
        ...
        ValueError: objects are not aligned

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return float(dot(arg_1, arg_2) ** self.degree)

    def __repr__(self):
        return 'HomogeneousPolynomialKernel(' + repr(self.degree) + ')'

    def __str___(self):
        return self.__repr__()


class GaussianKernel(Kernel):
    r"""
    Gaussian kernel inducing in the original space a superposition of
    gaussian bells.

    :param sigma: gaussian standard deviation.

    :type sigma: float

    EXAMPLES:

    A :class:`GaussianKernel` object is obtained in function of the
    corresponding standard deviation:

    >>> from yaplf.models.kernel import GaussianKernel
    >>> k = GaussianKernel(1)

    Only positive values can be used as standard deviation, as a
    :exc:`ValueError` is otherwise thrown:

    >>> GaussianKernel(-5)
    Traceback (most recent call last):
    ...
    ValueError: -5 is not usable as a gaussian standard deviation

    Arguments of a gaussian kernel are numeric list or tuples
    (possibily intertwined) having the same length, expressed as arguments of
    :meth:`compute`:

    >>> k.compute((1, 0, 1), (0, 0, 1))
    0.60653065971263342
    >>> k.compute([-3, 1, 0.5], [1, 1.2, -8])
    6.7308528542235046e-20
    >>> k.compute([-1, -4, 3.5], (1, 3.2, 6))
    3.2909994469653827e-14

    Specification of iterables having unequal length causes a :exc:`ValueError`
    to be thrown:

    >>> k.compute([-1, 3.5], (1, 3.2, 6))
    Traceback (most recent call last):
    ...
    ValueError: shape mismatch: objects cannot be broadcast to a single shape


    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, sigma=1):
        r"""
        See :class:`GaussianKernel` for full documentation.

        """

        Kernel.__init__(self)
        if sigma > 0:
            self.sigma = sigma
        else:
            raise ValueError(str(sigma) +
                ' is not usable as a gaussian standard deviation')

    def compute(self, arg_1, arg_2):
        r"""
        Compute the gaussian kernel between :obj:`arg_1` and :obj:`arg_2`,
        where the kernel value :math:`k(x_1, x_2)` is intended as the quantity
        :math:`\mathrm e^{-\frac{||x_1 - x_2||^2}{2 \simga^2}`, :math:`\sigma`
        being the kernel standard deviation.

        :param arg_1: first argument to the gaussian kernel.

        :param arg_2: second argument to the gaussian kernel.

        :returns: kernel value.

        :rtype: float

        EXAMPLES:

        Arguments of :meth:`compute` are numeric list or tuples (possibily
        intertwined) having the same length:

        >>> from yaplf.models.kernel import GaussianKernel
        >>> k = GaussianKernel(1)
        >>> k.compute((1, 0, 1), (0, 0, 1))
        0.60653065971263342
        >>> k.compute([-3, 1, 0.5], [1, 1.2, -8])
        6.7308528542235046e-20
        >>> k.compute([-1, -4, 3.5], (1, 3.2, 6))
        3.2909994469653827e-14

        Specification of iterables having unequal length causes a
        :exc:`ValueError` to be thrown:

        >>> k.compute([-1, 3.5], (1, 3.2, 6))
        Traceback (most recent call last):
        ...
        ValueError: shape mismatch: objects cannot be broadcast to a single
        shape

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """

        return float(exp(-1. * norm(array(arg_1) - array(arg_2)) ** 2 /\
            (2 * self.sigma ** 2)))

    def __repr__(self):
        return 'GaussianKernel(' + repr(self.sigma) + ')'

    def __str___(self):
        return self.__repr__()


class HyperbolicKernel(Kernel):
    r"""
    Hyperbolic kernel inducing in the original space a superposition of
    gaussian bells.

    :param scale: scale constant.

    :type scale: float

    :param offset: offset constant.

    :type offset: float

    EXAMPLES:

    A :class:`HyperbolicKernel` object is obtained in function of its degree:

    >>> from yaplf.models.kernel import HyperbolicKernel
    >>> k = HyperbolicKernel(1, 5)

    Arguments of a gaussian kernel are numeric list or tuples
    (possibily intertwined) having the same length, expressed as arguments of
    :meth:`compute`:

    >>> k.compute((1, 0, 1), (0, 0, 1))
    0.99998771165079559
    >>> k.compute([-3, 1, 0.5], [1, 1.2, -8])
    -0.66403677026784891
    >>> k.compute([-1, -4, 3.5], (1, 3.2, 6))
    0.99999999994938904

    Specification of iterables having unequal length causes a :exc:`ValueError`
    to be thrown.

    >>> k.compute([-1, 3.5], (1, 3.2, 6))
    Traceback (most recent call last):
    ...
    ValueError: matrices are not aligned

    AUTHORS:

    - Dario Malchiodi (2011-02-05)

    """

    def __init__(self, scale=1, offset=0):
        r"""
        See :class:`HyperbolicKernel` for full documentation.

        """

        Kernel.__init__(self)
        self.scale = scale
        self.offset = offset

    def compute(self, arg_1, arg_2):
        r"""
        Compute the hyperbolic kernel between :obj:`arg_1` and :obj:`arg_2`,
        where the kernel value :math:`k(x_1, x_2)` is intended as the quantity
        :math:`\tanh(k x_1 \dot x_2 + q)`, :math:`k` and :math:`q` being the
        scale and offset values, respectively.

        :param arg_1: first argument to the gaussian kernel.

        :param arg_2: second argument to the gaussian kernel.

        :returns: kernel value.

        :rtype: float

        EXAMPLES:

        Arguments of :meth:`compute` are numeric list or tuples (possibily
        intertwined) having the same length:

        >>> from yaplf.models.kernel import HyperbolicKernel
        >>> k = HyperbolicKernel(1, 5)
        >>> k.compute((1, 0, 1), (0, 0, 1))
        0.99998771165079559
        >>> k.compute([-3, 1, 0.5], [1, 1.2, -8])
        -0.66403677026784891
        >>> k.compute([-1, -4, 3.5], (1, 3.2, 6))
        0.99999999994938904

        Specification of iterables having unequal length causes a
        :exc:`ValueError` to be thrown:

        >>> k.compute([-1, 3.5], (1, 3.2, 6))
        Traceback (most recent call last):
        ...
        ValueError: matrices are not aligned

        AUTHORS:

        - Dario Malchiodi (2011-02-05)

        """

        return float(tanh(self.scale * dot(arg_1, arg_2) +  self.offset))

    def __repr__(self):
        return 'HyperbolicKernel(' + repr(self.scale) + ', ' + repr(self.offset) + ')'

    def __str___(self):
        return self.__repr__()


class PrecomputedKernel(Kernel):
    r"""
    Custom kernel whose entries are precomputed and stored in a matrix.

    :param kernel_computations: kernel computations.

    :type kernel_computations: square matrix of float elements

    EXAMPLES:

    A precomputed kernel is created through specification of a square matrix of
    numeric values. Subsequent invocations of :meth:`compute` should be based
    on integer arguments referring to indices in this matrix:

    >>> from yaplf.models.kernel import PrecomputedKernel
    >>> k = PrecomputedKernel(((9, 1, 4, 4), (1, 1, 1, 1), (4, 1, 4, 1), \
    ... (4, 1, 1, 4)))

    Specification of non-square matrices as arguments to the constructor cause
    a :exc:`ValueError` to be thrown:

    >>> PrecomputedKernel(((1, 2), (3, 4, 5)))
    Traceback (most recent call last):
    ...
    ValueError: The supplied matrix is not array-like or is not square

    Invocations of :meth:`compute` should specify as arguments two indices
    for row and column of the above mentioned matrix:

    >>> k.compute(1, 1)
    1.0
    >>> k.compute(0, 0)
    9.0

    When the specified indices are incompatible with the kernel matrix passed
    to the constructor, an :exc:`IndexError: is thrown:

    >>> k.compute(0, 10)
    ...
    IndexError: tuple index out of range

    Likewise, :exc:`TypeError` is thrown if non-integer values are specified:

    >>> k.compute(0, 1.6)
    ...
    TypeError: tuple indices must be integers, not float


    AUTHORS:

    - Dario Malchiodi (2010-02-22)

    """

    def __init__(self, kernel_computations):
        r"""
        See ``PrecomputedKernel`` for full documentation.

        """

        Kernel.__init__(self)
        self.precomputed = True
        try:
            (rows, columns) = shape(kernel_computations)
        except ValueError:
            raise ValueError('The supplied matrix is not array-like ' + \
                'or is not square')

        if rows != columns:
            raise ValueError('The supplied matrix is not square')

#        if rows != len(patterns):
#            raise ValueError('The supplied matrix is not compatible \
#                with the number of patterns')

        self.kernel_computations = kernel_computations
#        self.patterns = patterns

    def compute(self, arg_1, arg_2):
        r"""
        Recall the precomputed kernel value when ind_1 and ind_2 are
        indices corresponding to two patterns.

        INPUT:

        - ``arg_1`` -- first kernel argument.

        - ``arg_2`` -- second kernel argument.

        OUTPUT:

        float -- kernel value.

        EXAMPLES:

        Arguments of ``compute`` are integers corresponding to the original
        patterns in a sample:

        ::

            >>> from yaplf.models.kernel import PrecomputedKernel
            >>> k = PrecomputedKernel(((1, 2), (3, 4)))
            >>> k.compute(1, 1)
            4.0
            >>> k.compute(1, 0)
            3.0

        ::

        Specification of an invalid argument to the ``compute``method causes an
        IndexError to be thrown. For instance, the kernel previously defined
        has stored a `2 \times 2` matrix, so that only `0` and `1` will be
        valid arguments:

        ::

            >>> k.compute(1, 2)
            Traceback (most recent call last):
                ...
            IndexError: tuple index out of range

        AUTHORS:

        - Dario Malchiodi (2010-02-22)

        """


#        ind_1 = self.patterns.index(arg_1)
#        ind_2 = self.patterns.index(arg_2)
#        
#        return float(self.kernel_computations[ind_1][ind_2])
        return float(self.kernel_computations[arg_1][arg_2])

    def __repr__(self):
        return 'PrecomputedKernel(' + repr(self.kernel_computations) + ')'

    def __str___(self):
        return self.__repr__()
