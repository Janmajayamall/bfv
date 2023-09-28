use bfv::*;
use itertools::izip;
use rand::thread_rng;

fn decrypt_and_print(sk: &SecretKey, evaluator: &Evaluator, ct: &Ciphertext, m: &[u64]) {
    let res = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct), Encoding::default());
    println!("m: {:?}", m);
    println!("m after rotation: {:?}", res);
}

fn main() {
    // BFV ciphertext with `slots` can be viewed as matrix of dimension `2 x (slots/2)`. Upper row corresponds upper half of ciphertext slots and lower row corresponds to lower half of ciphertext slots.
    // We can rotate the row vectors right/left and swap the rows.
    // Warning: The parameters are not secure.
    let mut params = BfvParameters::new(&[50, 50, 50], 65537, 16);
    params.enable_hybrid_key_switching(&[50, 50, 50]);

    let mut rng = thread_rng();

    let sk = SecretKey::random_with_params(&params, &mut rng);

    let evaluator = Evaluator::new(params);

    // Randomly generate a message and encrypt it
    let m0 = evaluator
        .params()
        .plaintext_modulus_op
        .random_vec(16, &mut rng);
    let pt = evaluator.plaintext_encode(&m0, Encoding::default());
    let ct = evaluator.encrypt(&sk, &pt, &mut rng);

    // Generate evaluation key to rotate the ciphertext left by 1, right by 1, and row swap;
    let rotation_indices = [
        // left by 1
        1,
        // right by 1,
        -1,
        // row swap = 2 * slots - 1
        2 * 16 - 1,
    ];
    let ek = EvaluationKey::new(
        evaluator.params(),
        &sk,
        &[],
        &[0, 0, 0],
        &rotation_indices,
        &mut rng,
    );

    // rotate ciphertext left by 1
    let ct_rot = evaluator.rotate(&ct, 1, &ek);
    decrypt_and_print(&sk, &evaluator, &ct_rot, &m0);

    // rotate ciphertext right by 1
    let ct_rot = evaluator.rotate(&ct, -1, &ek);
    decrypt_and_print(&sk, &evaluator, &ct_rot, &m0);

    // row swap
    let ct_rot = evaluator.rotate(&ct, 2 * 16 - 1, &ek);
    decrypt_and_print(&sk, &evaluator, &ct_rot, &m0);
}
