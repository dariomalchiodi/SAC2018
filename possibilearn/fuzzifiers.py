import numpy as np

class BaseFuzzifier(object):
    def __init__(self):
        pass

        def get_fuzzified_membership(self, SV_square_distance, sample,\
                     estimated_square_distance_from_center):
            raise NotImplementedError(
            'the base class does not implement get_fuzzified_membership method')

class CrispFuzzifier(BaseFuzzifier):
    def __init__(self):
        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

    def get_fuzzified_membership(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            return 1 if r <= SV_square_distance else 0
        return estimated_membership

    def __repr__(self):
        return 'CrispFuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True


class LinearFuzzifier(BaseFuzzifier):
    def __init__(self):
        self.name = 'Linear'
        self.latex_name = '$\\hat\\mu_{\\text{lin}}$'

    def get_fuzzified_membership(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        max_radius = np.max(map(estimated_square_distance_from_center,
                                sample))

        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            result = (max_radius - r)/(max_radius - SV_square_distance)
            return 1 if r <= SV_square_distance \
                     else result if result > 0 \
                     else 0
        return estimated_membership

    def __repr__(self):
        return 'LinearFuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

class QuantileConstantPiecewiseFuzzifier(BaseFuzzifier):
    def __init__(self):
        self.name = 'QuantileConstPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{qconst}}$'

    def get_fuzzified_membership(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        sample = map(estimated_square_distance_from_center, sample)
        m = np.median([s-SV_square_distance
                       for s in sample if s > SV_square_distance])
        q1 = np.percentile([s-SV_square_distance
                            for s in sample if s > SV_square_distance], 25)
        q3 = np.percentile([s-SV_square_distance
                            for s in sample if s > SV_square_distance], 75)

        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            return 1 if r <= SV_square_distance \
                     else 0.75 if r <= SV_square_distance + q1 \
                     else 0.5 if r <= SV_square_distance + m \
                     else 0.25 if r <= SV_square_distance + q3 \
                     else 0
        return estimated_membership

    def __repr__(self):
        return 'QuantileConstantPiecewiseFuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

class QuantileLinearPiecewiseFuzzifier(BaseFuzzifier):
    def __init__(self):
        self.name = 'QuantileLinPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{qlin}}$'

    def get_fuzzified_membership(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        sample = map(estimated_square_distance_from_center, sample)
        m = np.median([s-SV_square_distance
                       for s in sample if s > SV_square_distance])
        q1 = np.percentile([s-SV_square_distance
                            for s in sample if s > SV_square_distance], 25)
        q3 = np.percentile([s-SV_square_distance
                            for s in sample if s > SV_square_distance], 75)
        mx = np.max(sample) - SV_square_distance

        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            ssd = SV_square_distance
            return 1 if r <= ssd \
                 else (-r + ssd)/(4*q1) + 1 if r <= ssd + q1 \
                 else (-r + ssd + q1)/(4*(m-q1)) + 3.0/4 if r <= ssd + m \
                 else (-r + ssd + m)/(4*(q3-m)) + 1./2 if r <= ssd + q3 \
                 else (-r + ssd + q3)/(4*(mx-q3)) + 1./4 if r <= ssd+mx \
                 else 0

        return estimated_membership

    def __repr__(self):
        return 'QuantileLinearPiecewiseFuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

class ExponentialFuzzifier(BaseFuzzifier):
    def __init__(self, alpha):
        self.alpha = alpha
        self.name = 'Exponential({})'.format(alpha)
        self.latex_name = '$\\hat\\mu_{{\\text{{exp}},{:.3f}}}$'.format(alpha)

    def get_fuzzified_membership(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        sample = map(estimated_square_distance_from_center, sample)

        q = np.percentile([s-SV_square_distance
                           for s in sample if s > SV_square_distance],
                          100*self.alpha)
        ssd = SV_square_distance
        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            return 1 if r <= ssd \
                     else np.exp(np.log(self.alpha)/q * (r - ssd))
        return estimated_membership

# class ExponentialFuzzifier(BaseFuzzifier):
#     ''' p is a percentile here! '''
#     def __init__(self, p):
#         self.p = p
#         self.name = 'Exponential({}%)'.format(p)
#
#     def get_fuzzified_membership(self, SV_square_distance, sample,
#                  estimated_square_distance_from_center):
#         sample = map(estimated_square_distance_from_center, sample)
#
#         q_p = np.percentile([s for s in sample if s > SV_square_distance], self.p)
#         ssd = SV_square_distance
#         def estimated_membership(x):
#             r = estimated_square_distance_from_center(np.array(x))
#             val = 1 if r <= ssd \
#                     else np.exp(np.log(self.p)/q_p * (r - ssd))
#             return val if val >= 0 else 0
#         return estimated_membership
#
#     def __repr__(self):
#         return 'ExponentialFuzzifier({})'.format(self.p)
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __eq__(self, other):
#         return type(self) == type(other) and self.p == other.p
#
#     def __ne__(self, other):
#         return not self == other
#
#     def __hash__(self):
#         return hash(self.__repr__())
#
#     def __nonzero__(self):
#         return True
