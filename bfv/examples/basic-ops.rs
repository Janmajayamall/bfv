use bfv::*;
use itertools::izip;
use rand::thread_rng;

fn decrypt_and_assert(sk: &SecretKey, evaluator: &Evaluator, ct: &Ciphertext, m: &[u64]) {
    let res = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct), Encoding::default());
    assert_eq!(res, m);
}

fn main() {
    // plaintext modulus
    let t = 65537;
    // no. of slots (ie degree)
    let slots = 1 << 14;

    let mut rng = thread_rng();

    // Parameters for 128 bit security
    // Q - 288 bits
    let mut params = BfvParameters::new(&[38, 50, 50, 50, 50, 50], t, slots);
    // enable hybird key switching
    // P - 150 bits
    params.enable_hybrid_key_switching(&[50; 3]);

    // generate secret key
    let sk = SecretKey::random_with_params(&params, &mut rng);

    // Create evaluator to evaluate arithmetic operarions
    let evaluator = Evaluator::new(params);

    // t modulus
    let plaintext_modulus = Modulus::new(t);

    macro_rules! correctness {
        ($m0:tt, $m1:tt, $f:tt, $cipher:tt) => {
            let mut m0_clone = $m0.clone();
            plaintext_modulus.$f(&mut m0_clone, &$m1);
            decrypt_and_assert(&sk, &evaluator, &$cipher, &m0_clone);
        };
    }

    // Generate random messages and encrypt them
    let m0 = plaintext_modulus.random_vec(slots, &mut rng);
    let m1 = plaintext_modulus.random_vec(slots, &mut rng);
    // encode as plaintexts
    // Choose Encoding::default() if the plaintext is intended to be encrypted
    let p0 = evaluator.plaintext_encode(&m0, Encoding::default());
    let p1 = evaluator.plaintext_encode(&m1, Encoding::default());
    // encrypt
    let ct0 = evaluator.encrypt(&sk, &p0, &mut rng);
    let ct1 = evaluator.encrypt(&sk, &p1, &mut rng);

    // Ciphertexts operations
    // Add
    let ct2 = evaluator.add(&ct0, &ct1);
    correctness!(m0, m1, add_mod_fast_vec, ct2);

    // Sub
    let ct2 = evaluator.sub(&ct0, &ct1);
    correctness!(m0, m1, sub_mod_fast_vec, ct2);

    // Multiply
    // mutliplication increase inrease the degree of ciphertext from 2 to 3. To further compute on the output we will have to relinearize the `ct2` (introduced later)
    let ct2 = evaluator.mul(&ct0, &ct1);
    correctness!(m0, m1, mul_mod_fast_vec, ct2);

    // Ciphertext and plaintext operations
    let m2 = plaintext_modulus.random_vec(slots, &mut rng);
    // Encoding::simd encodes plaintext such that it can be used for slot-wise arithematic.
    // Level indicates level of the ciphertext with which the plaintext is inteded to be used.
    // EncodingType
    // `PolyCache::AddSub(Representation)` encodes the plaintext such that it can be used for efficient
    // addition/subtration with the ciphertext. `Representation` must be set to `Representation::Coefficient` (usually the case)
    // if ciphertext polynomials are in `Representation::Coefficient`, otherwise set to `Representation::Evaluation`.
    // `PolyCache::AddSub(Representation)`
    // `PolyCache::Mul(PolyType::Q)` encodes the plaintext such that it used for efficient multiplication with the ciphertext.
    // `PolyType::Q` indicates that it is inteded to be multiplied with ciphertext polynomials, which might not always be the case.
    let pt_to_add = evaluator.plaintext_encode(
        &m2,
        Encoding::simd(0, PolyCache::AddSub(Representation::Coefficient)),
    );
    let ct2 = evaluator.add_plaintext(&ct0, &pt_to_add);
    correctness!(m0, m2, add_mod_fast_vec, ct2);

    let pt_to_multiply =
        evaluator.plaintext_encode(&m2, Encoding::simd(0, PolyCache::Mul(PolyType::Q)));
    // Plaintext multiplication is only allowed in `Representation::Evaluation` reprsenation
    let mut ct0_clone = ct0.clone();
    evaluator.ciphertext_change_representation(&mut ct0_clone, Representation::Evaluation);
    let ct2 = evaluator.mul_plaintext(&ct0_clone, &pt_to_multiply);
    correctness!(m0, m2, mul_mod_fast_vec, ct2);

    // Rotations
    // It is possible to rotate ciphertext slots left/right. BFV ciphertext with slots can be viewed as a mtrix of two rows where the first row corresponds to upper half slots and the second row corresponds to lower half slots.
    // You can rotate values in each of the rows and can swap the rows.
    // For rotations we will first need to generate EvaluationKey.
    // Generate `EvaluationKey` to rotate left by 1 with ciphertext at level 0.
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[1], &mut rng);
    // rotate left by 1
    let ct2 = evaluator.rotate(&ct0, 1, &ek);
    // Generate `EvaluationKey` to row swap
    // To row swap pass in `degree * 2 - 1` (TODO: improve API to generate evaluation keys for row swap)
    let row_swap_index = evaluator.params().degree * 2 - 1;
    let ek = EvaluationKey::new(
        evaluator.params(),
        &sk,
        &[0],
        &[0],
        &[row_swap_index as isize],
        &mut rng,
    );
    let ct2 = evaluator.rotate(&ct0, row_swap_index as isize, &ek);

    // Levels and repeated multiplications

    // To operate on ciphertext after ciphertext multiplication we must relienarise the ciphertext to reduce it from degree 3 to degree 2.
    // To relienarize we additionally require evaluation key that contains relinerization key at ciphertext's level
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);
    let ct2 = evaluator.mul(&ct0, &ct1);
    let ct2_relin = evaluator.relinearize(&ct2, &ek);
    assert_eq!(ct2.c_ref().len(), 3);
    assert_eq!(ct2_relin.c_ref().len(), 2);

    // Every operstion on ciphertext increases the noise in the resulting ciphertext and the magnitude of noise growth depends on the operation. Add/Sub/Rotations operations cause small noise growth where as multiplications cause higher noise growth. A ciphertext becomes useless when its noise reaches a certain threshold (ie noise flows into plaintext bits) and decrypting it will result in incorrect decryption. Thus, given BFV parameters you can only perform limited no. of ciphertext operations. Since multiplications cause higher noise growth, the no. of possible FHE computation is often dictated by no. of ciphertext mutliplications.
    // Notice that the noise in ciphertext after multiplication, ct2, is higher than `ct0` and `ct1`
    println!(
        "Noise of ciphertext after multiplicaiton: {}",
        evaluator.measure_noise(&sk, &ct2)
    );
    println!(
        "Noise of input ciphertexts ct0: {}, ct1: {}",
        evaluator.measure_noise(&sk, &ct0),
        evaluator.measure_noise(&sk, &ct1)
    );
    assert_eq!(
        evaluator.measure_noise(&sk, &ct2),
        evaluator.measure_noise(&sk, &ct2_relin)
    );

    // Ciphertext at hgiher levels are more expensive to operate on than ciphertext at lower levels. For ex, ciphertexts at level 0 require more time to operate on than ciphertexts at level 1 (yes, level 0 is higher than level 1 and level 2 is higher than level 1 and so on so forth).
    // Once we have accumulated enough noise it is always a nice idea to drop down the level to improve performance of further ciphertext operations. For example, if the noise in ciphertext is 70 and ciphertext modulus is set to [50,50,50] then we can drop ciphertext from level 0 to level 1 to
    // get rid of 50 bits of noise.
    // Generate relinearization key for level 0
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);
    // let's calculate ct0^4 (ie multiply 2 times)
    let ct0_sq = evaluator.relinearize(&evaluator.mul(&ct0, &ct0), &ek);
    println!(
        "Noise after ct0*ct0: {}",
        evaluator.measure_noise(&sk, &ct0_sq)
    );
    let mut ct0_sq_sq = evaluator.relinearize(&evaluator.mul(&ct0_sq, &ct0_sq), &ek);
    println!(
        "Noise after (ct0*ct0)*(ct0*ct0): {}",
        evaluator.measure_noise(&sk, &ct0_sq_sq)
    );
    // we can get rid of accumulated noise after two ciphertext multiplications by levelling down
    evaluator.mod_down_next(&mut ct0_sq_sq);
    println!(
        "Noise in (ct0*ct0)*(ct0*ct0) after levelling down once: {}",
        evaluator.measure_noise(&sk, &ct0_sq_sq)
    );

    // mul lazy
    // Ciperhtext by ciphertext multiplication is an expensive operation and consists of (1) expanding basis from Q to PQ (2) Tensor (3) Scaling down and switching basis from PQ to Q. In situations where you need to multiply several ciphertexts and add them together you can avoid step (3) after each ciphertext multiplication and do it only once after adding the results of tensoring.
    // This improves runtime without cuasing additional error growth.
    // Create 2 vectors of ciphertexts
    let ct0_vector = vec![ct0.clone(); 10];
    let ct1_vector = vec![ct1.clone(); 10];
    // We will calculate hadamard product of the ct0_vector and ct1_vector and add the results into a single ciphertext (ie inner product). To multiply instead of `mul` we will use `mul_lazy` to skip (3) after each multiplication and
    // only do (3) once after adding all the products
    let mut ct_sum = Ciphertext::placeholder();
    izip!(ct0_vector.iter(), ct1_vector.iter()).for_each(|(c0, c1)| {
        let r = evaluator.mul_lazy(c0, c1);
        // For 0^th iteration ct_sum will be a placeholder (ie ciphertext will be empty)
        if ct_sum.c_ref().len() == 0 {
            ct_sum = r;
        } else {
            evaluator.add_assign(&mut ct_sum, &r);
        }
    });
    // Perform (3) only once
    let ct_sum = evaluator.scale_and_round(&mut ct_sum);
    {
        let mut m0_clone = m0.clone();
        plaintext_modulus.mul_mod_fast_vec(&mut m0_clone, &m1);

        // multiply by scalar 10
        plaintext_modulus.scalar_mul_mod_fast_vec(&mut m0_clone, 10);
        decrypt_and_assert(&sk, &evaluator, &ct_sum, &m0_clone);
    }
}
