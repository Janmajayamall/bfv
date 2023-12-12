We implement multi-party operations for public key generation, relinearization key generation, galois key generation, and collective key switching as outlined in paper - https://eprint.iacr.org/2020/304.pdf.

Note: All communications between the parties require an authenticated, not necessarily private, channels. So one can assume each transmission as a broadcast operation.

**Collective public key generation:**

Collective public key generation is a single round protocol and outputs the collective public key.

1. Parties agree upon a random common reference string: `crs`.
2. Each party generates their share using `CollectivePublicKeyGenerator::generate_share`
3. Each party broadcasts their share. 
4. Each party individually constructs the public key by calling `CollectivePublicKeyGenerator::aggregate_shares_and_finalise` with received, along with their own, shares. 

**Collective relinearization key generation:**

Relinearization key is required to reduce the degree of ciphertext from 3 to 2. Given output of ciphertext multiplication $(c_0, c_1, c_2)$, relinearization homomorphically multiplies $c_2 \cdot s_{ideal}^2$ and outputs $(c'_0, c'_1)$ such that $c'_0 + c'_1s_{ideal} = c_2s^2_{ideal}$.

Collective relinearization key generation is the only 2 round protocol. 

1. Parties agree upon a random common reference string: `crs`
2. Each party instantiates their state for the 2 party protocol by calling `CollectiveRlkGenerator::init_state`. State only stores a ephemeral secret for the duration of the key generation protocol.
3. Each party generates their $share_1$ by calling `CollectiveRlkGenerator::generate_share_1`.
4. Parties broadcast their generated $share_1$s.
5. Each party aggregates all received $share_1$s by calling `CollectiveRlkGenerator::aggregate_shares_1`.
6. With aggregated $share_1$s each party starts round 2 of the protocol. Each party generates their $share_2$ by calling `CollectiveRlkGenerator::generate_share_2`.
7. Parties broadcast their generated $share_2$s.
8. Each party aggregates received $share_2$s by calling `CollectiveRlkGenerator::aggregate_shares_2`. `CollectiveRlkGenerator::aggregate_shares_2` also finalises the protocol by constructing the relinearization key.

**Collective decryption**

Given ciphertext $ct$ encrypted using collective public key, collective decryption is a single round protocol that decrypts $ct$.

Note: Collective decryption is special case of collective key switch outlined as Protocol 3 in https://eprint.iacr.org/2020/304.pdf where $s'$ is set to zero polynomial.

To decrypt ciphertext $ct$:
1. Each party generates their ciphertext dependent share by calling `CollectiveDecryption::generate_share`.
2. Parties broadcast their share.
3. Each party aggregates the received shares and decrypts the ciphertext by calling `CollectiveDecryption::aggregate_share_and_decrypt`.





