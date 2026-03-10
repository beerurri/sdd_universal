use crate::constants;
use crate::params;
use nalgebra::DMatrix;
use rand::Rng;
use rand::rngs::StdRng;


pub fn perform_testing(node_states: &Vec<f64>, target_data: &Vec<f64>, params: &params::Params) -> f64 {
    // node states matrix of size set_len x nodes_count (5000x10+1)
    let slice = &node_states.iter().step_by(params.NODES / params.TRAINING_NODES).copied().collect::<Vec<f64>>();
    println!("NODES_COUNT = {}", slice.len());

    let mut node_matrix = DMatrix::from_row_slice(
        constants::SAMPLES_COUNT as usize,
        params.TRAINING_NODES,
        slice
    );
    node_matrix = node_matrix.insert_column(params.TRAINING_NODES, 1.);
    
    let target_matrix = DMatrix::from_row_slice(
        constants::SAMPLES_COUNT,
        constants::FEATURES_OUT,
        &target_data
    );

    let nodes_train = node_matrix.rows(
        constants::WARM_UP_COUNT,
        constants::NARMA_TRAIN_SET_LEN
    );
    let target_train = target_matrix.rows(
        constants::WARM_UP_COUNT,
        constants::NARMA_TRAIN_SET_LEN
    );
    
    let nodes_test = node_matrix.rows(
        constants::WARM_UP_COUNT + constants::NARMA_TRAIN_SET_LEN,
        constants::NARMA_TEST_SET_LEN
    );
    let target_test = target_matrix.rows(
        constants::WARM_UP_COUNT + constants::NARMA_TRAIN_SET_LEN,
        constants::NARMA_TEST_SET_LEN
    );

    let w =
        (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
        * &nodes_train.transpose() * &target_train;

    let predicted = nodes_test * &w;

    // let mut nses: Vec<f64> = Vec::with_capacity(constants::NARMA_TEST_SET_LEN);
    let mut ses: Vec<f64> = Vec::with_capacity(constants::NARMA_TEST_SET_LEN);
    let variance = stat::variance(&target_test.column(0).iter().copied().collect::<Vec<f64>>());
    // println!("variance: {}", variance);

    for i in 0..constants::NARMA_TEST_SET_LEN {
        let tmp_predicted: f64 = predicted[(i, 0)];
        let tmp_target: f64 = target_test[(i, 0)];

        // println!("tmp_predicted: {}, tmp_target: {}", tmp_predicted, tmp_target);

        // let nse = (tmp_target - tmp_predicted).powi(2) / variance;
        let se = (tmp_target - tmp_predicted).powi(2);

        // nses.push(nse);
        ses.push(se);
    }

    // println!("{}", nses.iter().sum::<f64>());
    println!("{}", ses.iter().sum::<f64>());

    // let nmse: f64 = nses.iter().sum::<f64>() / constants::NARMA_TEST_SET_LEN as f64;
    let mse = ses.iter().sum::<f64>() / constants::NARMA_TEST_SET_LEN as f64;
    let nrmse = (mse / variance).sqrt();

    // println!("mean_nmse: {}", nmse);

    return nrmse;
}


pub fn make_narma10_data(rng: &mut StdRng, count: usize) -> (Vec<f64>, Vec<f64>) {
    let u: Vec<f64> = (0..count)
        .map(|_| rng.random_range(0.0..=0.5))
        .collect();
    let mut y: Vec<f64> = vec![0.0; count];

    for k in 10..(count - 1) {
        y[k+1] = 0.3 * y[k] +
                0.05 * y[k] * y[(k-9)..k].iter().sum::<f64>() +
                1.5 * u[k] * u[k-9] +
                0.1;
    }

    return (u, y);
}