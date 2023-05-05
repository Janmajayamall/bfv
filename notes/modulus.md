# Barrett Modulus reduction

We can compute $x \mod m$ in the following way:
$y = x - (q \times m), q = \lfloor \frac{x}{m} \rfloor $

However computing $q$ requires integer division which is expensive in computers. Barrett modulus reduction is a way to avoid integer division. Instead it uses pre-computed inverse of $m$ multiplied by some factor and estimates $q$ using only shifts and multiplication.

Algorithm 2 of https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf specifies generalised barrett modulus reduction for integers.

### $<64 + \gamma$ bits by $<64$ bits modulus reduction

1. $m$ is the modulus. $n = \log_2m$.
2. To reduce $x$ where $\log_2x <= n + \gamma$ by modulus $m$ with $n$ bits, where $\gamma < n$, we implement Algorithm 2 of https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf.
3. $ \alpha = n + 3 $ and $ \beta = -2 $. Generally $\alpha \ge  \gamma + 1$ where $\gamma + n \ge log_{2} x$ and $x$ is dividend in $\lfloor \frac{x}{m} \rfloor$.
4. $\mu = \frac{2^{\alpha + n}}{m} = \frac{2^{2n + 3}}{m} $
5. $\mu$ is stored in 64 bits. Since $\log_2mu = (2n+3) - n = n+3$, $\mu$ will only fit in 64 bits if $n < 61$. Hence, implementation will given incorrect result for $n \ge 61$.
6. $\alpha = n+3$ and $\beta = -2$ play an important role in estimation of $q = \lfloor \frac {x}{m} \rfloor$. I have observed with set $\alpha \& \beta$ values that as long as $log_2x < 2n$ estimation of $q$ is accurate such that you do not pay for correction cost.
7. Generally if $log_2x \ge 2n$ the result will still be correct however you will have to pay for a correction cost. Although, loop iteration is limited to 1-2 times.
8. $log_2x$ cannot be arbitrarily big due to 128 bit limitation of implementation. To be specific, when we multiple $ab >> (n-2)$ by $\mu$ we must satisfy following in-equality to prevent overflow: $log_2x - (n-2) + \log_2mu < 128$. Since $log_2mu = 2n + 3 - n = n + 3$, we can write in-equality as $log_2x + 5 < 128$. Hence, the result should be correct as long as $log_2x < 123$.

### 128 bits by 64 bits modulus reduction

1. We want to reduce a value that is `2n` bits by modulus that is `n` bits. The implementation follows Algorithm 14.42 of https://cacr.uwaterloo.ca/hac/about/chap14.pdf and https://github.com/openfheorg/openfhe-development/blob/055c89778d3d0dad00479150a053124137f4c3ca/src/core/include/utils/utilities-int.h#L59.

### When to use which?
