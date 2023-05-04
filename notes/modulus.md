# Barrett Modulus reduction

We can compute x mod m in the following way:
y = x - (q \* m), where q = floor(x / m)

However computing `q` requires integer division which is expensive in computers. Barrett modulus reduction is a way to avoid having to calculate `q` pre-computing inverse of m multiplied by some factor and then using only shifts and multiplication to estimate `q` as close as possible. Note that correctness of estimation depends on \alpha and \beta values.

Algorithm 2 of https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf specifies generalised barrett modulus reduction for integers.

### 64 bit by 64 bit modulus reduction

1. We want to reduce a value that is `n + \gamma` bits by modulus that is `n` bits, where `\gamma < n`. We implement Algorithm 2 of https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf.
2. \alpha = n + 3. Generally \alpha is \ge to \gamma + 1 where \gamma + \n are number of bits in `x` (ie dividend). Since `x` is limited by 64 bits, by setting \alpha = 3 and assuming n = 60 we have taken the upper bound.
3. \beta = -2
4. \mu = 2^(2n + 3) / m. Notice that \mu can fit in 64 bits if we assume that n is 60 bits (which is usually the case).
5. The implementation follows the implementation [here](https://github.com/openfheorg/openfhe-development/blob/c48c41cf7893feb94f09c7d95284a36145ec0d5e/src/core/include/math/hal/intnat/ubintnat.h#L1417). Read their comment to understand the choce of \alpha and \beta.

### 128 bit by 64 bit modulus reduction

1. We want to reduce a value that is `2n` bits by modulus that is `n` bits. The implementation follows Algorithm 14.42 of https://cacr.uwaterloo.ca/hac/about/chap14.pdf and https://github.com/openfheorg/openfhe-development/blob/055c89778d3d0dad00479150a053124137f4c3ca/src/core/include/utils/utilities-int.h#L59.
