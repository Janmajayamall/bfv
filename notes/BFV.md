
# RLWE

Search LWE 

Let's parameterise RLWE with a power of 2 $N$, standard deviation $\sigma$, and $Q \in Z^+$. 

Also note
1. $R_Q = Z_Q / (X^N + 1)$. Where $X^N + 1$  is $2N^{th}$ cyclotimic polynomial. This implies $R_Q$ is a cyclotomic field, thus a polynomial in $R_Q$ can be viewed as a polynomial in $R = Z/(X^N + 1)$ with its coefficients reduced $\mod Q$. 
2. $\chi_{e}$ denotes sampling randomly a polynomial $\in R_Q$ from gaussian distribution with standard deviation $\sigma$. Moreover, since $N$ is power of 2 sampling $R_Q$ from gaussian distribution corresponds to sampling each coefficient of polynomial from gaussian distribution independently. 
3. $\chi_{k}$ denotes a polynomial in $R_Q$ with its coefficients sampled randomly from ternary distribution, ie {-1,0,1}.

Randomly sample $A \in R_Q$, sample $e \in \chi_{e}$, and some secret polynomial $s \in \chi_k$. Set $B$ as:
$$B = As + e$$
Search-RLWE states that given $(B, A)$ it is hard to recover $s$. 

TODO: add Decisional RLWE

# BFV (RLWE) encryption

The moment cryptographers spot a hard problem, they try to figure out of a way to hide data using it. Since RLWE is conjectured to be hard, we can encrypt some data using it. 

To encrypt, first choose a plaintext space $t < Q$. Note that in practice, $t$ is way smaller than $Q$. Now let message be a polynomial $m$ in $R_Q$ with its coefficients reduced $\mod t$. 

We can hide message $m$ (not encrypt) as: 
$$B = As + e + m$$
Since RLWE is hard, recovering $m$ when only given $(B, -A)$ must be hard as well. However, in this case, even with access to $s$ we cannot recover $m$ since $m$ gets jumbled up with $e$. Thus, we need to scale $m$ by some factor and then add. 

Let's denote the scale factor with $\Delta$ and let $\Delta$ be:
$$\Delta = Q/t$$

We can encrypt $m$ as: 
$$B = As + e + \Delta m$$
Notice that, given $e$ is sampled from gaussian distribution and is small (small is a bit ambiguous here. It means that norm of $e$ is small. Note that there are two ways calculate norm, (1) Coefficient embedding (2) Canonical embedding. Papers use either of the embedding to derive formulas for noise growth). This time message bits (ie bits of coefficients of polynomial m) do not get jumbled up with bits of $e$. This is because bits of $m$ are stored in MSB. 

Given $s$ and $(B, -A)$, we can decrypt $(B, -A)$ as: 
$$\Delta m + e = B - As$$
Then we can remove the scaling factor $\Delta$ to find $m$ as: 
$$round(\frac{t (\Delta m + e)}{Q}) = m$$
Now since, 
$$round(\frac{t (\Delta m + e)}{Q}) = round(\frac{t\Delta m}{Q}) + round(\frac{te}{Q})$$
decryption will be correct only when 
$$round(\frac{te}{Q}) = 0$$
This will be the case as long as
$$e < \frac{Q}{2t}$$

Note that encoding messages in MSB is specific to BFV. Other FHE schemes may use other encoding technique. For ex, BGV encodes message in LSB. However, it is important to highlight that most FHE schemes rely on RLWE hardness. 

# Improvised BFV encryption

In practice, only approximate value of $\Delta$ can be calculated. Usually it is set to $floor(Q/t)$ and this introduces additional error to the ciphertext and affects decryption procedure. 

Let $r_t(Q) = Q - t\Delta$, that is the approximation introduced by flooring. This increases approximation error induced when removing the scaling factor during decryption. During decryption the equation now becomes: 
$$m - \frac{r_t(Q)m}{Q} + \frac{te}{Q}$$

This is because $\Delta = \frac{Q - r_t(Q)}{t}$ instead of $\frac{Q}{t}$.

This causes us to reserve more space for error. To be specific, now $e$ must satisfy the following inequality:
$$e < \frac{Q}{2t} - \frac{r_t(Q)}{2}$$

Also note that, another consequence of $r_t(Q)$ is higher noise growth after first multiplication. 

2021/204 introduced improvised encryption algorithm that sets $r_t(Q) \approx 0$. Improvised encryption algorithm scales $m$ by calculating 
$$round(\frac{Qm}{t})$$
instead of 
$$\Delta m, \space where \space \Delta = floor(Q/t)$$
Thus, using improvised encryption algorithm we can reduce the reserved space for error. $e$ must now satisfy
$$e < \frac{Q}{2t} - \frac{1}{2}$$
Additional $\frac{1}{2}$ is caused due to rounding errors in $round(\frac{Qm}{t})$. 

Remark 3.1 of 2021/204 shows how to calculate $round(\frac{Qm}{t})$ directly in RNS as long as $Q$ and $t$ are co-prime (usually they are). 

# Public Key BFV

You will notice that
$$B = As + e + \Delta m$$
requires secret key $s$ for encryption. For a public key encryption scheme we require a way to encrypt message to someone without their secret key. 

**Zero encryption as public key**

Public key $pk$ is set to: 
$$pk = (B, -A) = (As + e, -A)$$
$pk$ is simply a zero encryption under secret $s$.

**Using ephemeral key for encryption using $pk$**
Given $pk = (pk_0, pk_1)$, sample randomly $u \in \chi_k$ and $e_0,e_1 \in \chi_e$ and encrypt message $m$ as:
$$ct = (ct_0, ct_1) = (\Delta m + u \cdot pk_0 + e_0, u \cdot pk_1 + e_1)$$

Notice that both $ct_0$ and $ct_1$ are RLWE sample under ephemeral secret $u$. This implies public key encryption (1) is secure and (2) completely hides information about the intended recipient to which $pk$ belongs. 

It is vital to not reuse $u$ during public key encryption, otherwise you will risk revealing the $pk$ of intended recipient by publishing more than intended no. of RLWE samples for given security parameter. 

**Decryption**

Given access to $s$ and $ct$, decryption is:
$$ct_0 + ct_1 \cdot s = \Delta m + u\cdot pk_0 + e_0 + s\cdot u\cdot pk_1 + s\cdot e_1$$
$$= \Delta m + uAs + ue + e_0 - uAs + se_1$$
$$= \Delta m + ue + e_0 + se_1$$
Notice that noise in fresh ciphertext now becomes $v = ue + e_0 + se_1$. We should further scale down by $t/Q$ to find $m$:
$$m = round(\frac{t (\Delta m + v)}{Q})$$
We will get $m$ after scaling down as long as $e < \frac{Q}{2t} - \frac{1}{2}$.

Note: If you are wondering why use ephemeral keys, you must notice that it is required (& probably simplest way) to achieve property of hiding $pk$. Trivially adding $\Delta m$ to $pk$, that is $ct = (pk_0 + \Delta m, pk_1)$ will reveal $pk_1$, thus identity of to whom the $ct$ is encrypted. 

**Reducing noise in ciphertext after encrypting with public key**

Notice that noise in fresh ciphertext is higher, equal to $v$, when encrypted using $pk$ instead of $sk$. Given $Q$ there's no option to get rid of $v$ without sacrificing security. However, in case of hybrid key switching (or GHS variant of key switching) we measure security using modulus $PQ$ instead of just $Q$. The trick to reduce $v$ relies on setting $pk$ modulus $PQ$ and then calculate (ie mod switch) $[\frac{Q}{QP}ct]_Q$, thus reducing the noise by factor of $P$. 


