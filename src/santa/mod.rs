use crate::constants;
use crate::params;
use nalgebra::DMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};


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
        constants::SANTA_TRAIN_SET_LEN
    );
    let target_train = target_matrix.rows(
        constants::WARM_UP_COUNT,
        constants::SANTA_TRAIN_SET_LEN
    );
    
    let nodes_test = node_matrix.rows(
        constants::WARM_UP_COUNT + constants::SANTA_TRAIN_SET_LEN + constants::SANTA_GAP_SET_LEN,
        constants::SANTA_TEST_SET_LEN
    );
    let target_test = target_matrix.rows(
        constants::WARM_UP_COUNT + constants::SANTA_TRAIN_SET_LEN + constants::SANTA_GAP_SET_LEN,
        constants::SANTA_TEST_SET_LEN
    );

    let w =
        (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
        * &nodes_train.transpose() * &target_train;

    let predicted = nodes_test * &w;

    let mut nses: Vec<f64> = Vec::with_capacity(constants::SANTA_TEST_SET_LEN);
    let variance = stat::variance(&target_test.column(0).iter().copied().collect::<Vec<f64>>());
    // println!("variance: {}", variance);

    for i in 0..constants::SANTA_TEST_SET_LEN {
        let tmp_predicted: f64 = predicted[(i, 0)];
        let tmp_target: f64 = target_test[(i, 0)];

        // println!("tmp_predicted: {}, tmp_target: {}", tmp_predicted, tmp_target);

        let nse = (tmp_target - tmp_predicted).powi(2) / variance;

        nses.push(nse);
    }

    println!("{}", nses.iter().sum::<f64>());

    let nmse: f64 = nses.iter().sum::<f64>() / constants::SANTA_TEST_SET_LEN as f64;

    // println!("mean_nmse: {}", nmse);

    return nmse;
}


pub fn make_santa_data() -> (Vec<f64>, Vec<f64>) {
    let file = File::open("input_data.dat").unwrap();
    let reader = BufReader::new(file);

    let mut numbers: Vec<f64> = reader
        .lines()
        .filter_map(|line| line.unwrap().trim().parse::<f64>().ok())
        .collect();

    let min= numbers.iter().cloned().reduce(f64::min).unwrap();
    let max = numbers.iter().cloned().reduce(f64::max).unwrap();

    numbers = numbers.iter().map(|&x| (x - min) / max).collect();

    return (numbers[0..constants::SAMPLES_COUNT].to_vec(), numbers[constants::SANTA_K_STEPS..(constants::SAMPLES_COUNT+constants::SANTA_K_STEPS)].to_vec());
}