# BFV public key

For Public key encryption one should generate public key corresponding to a secret key. In (R)LWE based schemes, like BFV, ciphertexts encrypted using public may have higher error than ciphertexts encrypted using secret key. However, since we use hybrid variant of key switching we can eventually scale down error in fresh ciphertexts without any loss of security or efficiency.

## Generate public key

Public key equals zero encryption using the secret key. Given secret key $sk \in B_{N, Q}$, corresponding public key is tuple: 
$$pk = (pk_0, pk_1) = (a\cdot sk + e_0, -a) \in (R_{N,Q}, R_{N,Q})$$
where $a \in R_{N,Q}$, $e \in \chi_{N , Q}$.

## Encoding message as plaintext

To encode message $[m]_t$ as plaintext, one must calculate $[round(\frac{Q[m]_t}{t})]_Q$. Following remark 3.1 of 2021/204, if $Q$ and $t$ are co-prime it is possible to calculate $round(\frac{Q[m]_t}{t})$ in RNS directly. 

Observer that, $$[round(\frac{Q[m]_t}{t})]_Q = [\frac{Q[m]_t - [Qm]_t}{t}]_Q$$
Above equation holds true because Q are t are co-prime thus $Q[m]_t/t$ is not well defined $\mod Q$. Therefore, one should find closest value to $Q$ which is divisible by $t$. Since $[Qm]_t$ is remainder of $Qm/t$, subtracting $[Qm]_t$ from $Q[m]_t$ will give the closest representation to $Q[m]_t$ that is divisible by t.

Now notice that $Q[m]_t$ vanishes modulus $Q$. Thus we are left to calculate, 
$$[\frac{-[Qm]_t}{t}]_Q$$
This can be done easily in RNS by first scaling $m$ by $Q$ $\mod{t}$ and then multiplying $[Qm]_t$ with $[(-t)^{-1}]_Q$ as $[[Qm]_t\cdot[(-t)^{-1}]_Q]_Q$.

## Public key encryption

To encrypt encoded message $\Delta m \in R_{N,t}$, first sample a ephemeral secret key $u \in B_{N,Q}$ and let ciphertext equal tuple: 
$$(\Delta m + u \cdot pk_0 + e_1, u \cdot pk_1 + e_2) \in (R_{N,Q},R_{N,Q})$$
where $e_1,e_2 \in \chi_{N, Q}$.


## High error compared to secret key encryption

Given ciphertext, 
$$ct = (c_0, c_1) = (\Delta m + u \cdot pk_0 + e_1, u \cdot pk_1 + e_2) \in (R_{N,Q},R_{N,Q})$$

which is encrypted using
$$pk = (pk_0, pk_1) = (a\cdot sk + e_0, -a)$$

one with access to $sk$ can decrypt $ct$ as, 
$$c_0 + sk \cdot c_1$$
$$\Delta m + u (a \cdot sk) + u \cdot e_0 + e_1 + sk\cdot (-ua + e_2)$$
$$\Delta m + ue_0 + e_1 + sk \cdot e_2$$
The error term equals $ue_0 + e_1 + sk \cdot e_2$ and is greater than error term in fresh ciphertext encrypted using secret key. 


## Mod down to reduce error

Algorithm 2 of https://eprint.iacr.org/2018/931.pdf gives mod down function as: 
$$ModDown([X]_{Qr}) = [\frac{1}{r}X]_{Q}$$
where $Qr = q_0 \cdot q_1 \cdot ... \cdot r$

Given ciphertext $(c_0, c_1)$, let 
$$[c_0+ c_1\cdot s]_{Qr} = [\frac{Qr\cdot[m]_t}{t} + e]_{Qr}$$
Now notice that, 
$$ModDown([c_i]_{Qr}) = [\frac{1}{r}c_i]_{Q}$$
and,
$$[\frac{1}{r}(c_0+ c_1\cdot s)]_{Q} = [\frac{1}{r}(\frac{Qr\cdot[m]_t}{t} + e)]_{Q} = [\frac{Q[m]_t}{t} + \frac{e}{r}]_Q$$
Thus, the error is scaled down by $\log{r}$ bits. 

## PKE using union modulus Qr to reduce error in fresh ciphertexts

Follow the same procedure as before to generate public key and public key encryption but with ciphertext modulus set to $R_{Qr}$ instead of $R_Q$, where Qr is union of Q and another prime r.

Call function $ModDown()$ on polynomials of fresh ciphertext $(c_0, c1) \in (R_{Qr}, R_{Qr})$ to produce ciphertext $(c'_0, c'_1) \in (R_{Q}, R_{Q})$ with error scaled down by $\log{r}$ bits.

Using $Qr$ instead of $Q$ where $Q \cdot r = Qr$ causes security to decrease for a fixed FHE circuit. However, when using Hybrid (or GHS) variant of key switching one can use $Qr$ without security loss. This is because the security is defined by ciphertext modulus Q + SpecialP (i.e. special primes p) and SpecialP > r. Moreover a single prime suffices for $r$ and in the implementation we use the greatest prime from SpecialP moduli as $r$ (although doing so isn't necessary).


