use crate::params;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use crate::constants;
use nalgebra::DMatrix;
use statrs;
use statrs::distribution::ContinuousCDF;
use std::collections::HashSet;

pub fn perform_testing(input_data: &Vec<f64>, node_states: &Vec<f64>, rng: &mut StdRng, params: &mut params::Params) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // node states matrix of size set_len x nodes_count (5000x10+1)
    let slice = &node_states.iter().step_by(params.NODES / params.TRAINING_NODES).copied().collect::<Vec<f64>>();
    println!("NODES_COUNT = {}", slice.len());

    let mut node_matrix = DMatrix::from_row_slice(
        constants::SAMPLES_COUNT as usize,
        params.TRAINING_NODES,
        slice
    );
    node_matrix = node_matrix.insert_column(params.TRAINING_NODES, 1.);

    // input set matrix of size set_len x features_out=1 (5000x1)
    let input_matrix = DMatrix::from_row_slice(
        constants::SAMPLES_COUNT,
        constants::FEATURES_OUT,
        &input_data
    );

    let (mut lc_mean_through_mask, mut qc_mean_through_mask) = perform_testing_lc_qc(&node_matrix, &input_matrix, rng, params);

    let mut cc_mean_through_mask = perform_testing_cc(&node_matrix, &input_matrix, rng, params);
    
    let chi2 = statrs::distribution::ChiSquared::new(params.NODES as f64).unwrap();
    let p_val = 1e-4;
    let mc_thr = 2. * chi2.inverse_cdf(1. - p_val) / constants::SAMPLES_COUNT as f64;

    println!("\n\tnodes: {}\n\tmc_thr: {:.6}\n", params.NODES, mc_thr);

    let lc_d_threshold_idx = threshold_idx(
        &ma(&lc_mean_through_mask, 4),
        mc_thr
    );
    let lc_d_steps_threshold = sum_left_slice(
        &lc_mean_through_mask,
        lc_d_threshold_idx
    );
    lc_mean_through_mask.truncate(lc_d_threshold_idx);

    let qc_d_threshold_idx = threshold_idx(
        &ma(&qc_mean_through_mask, 4),
       mc_thr
    );
    let qc_d_steps_threshold = sum_left_slice(
        &qc_mean_through_mask,
        qc_d_threshold_idx
    );
    qc_mean_through_mask.truncate(qc_d_threshold_idx);

    let cc_d_threshold_idx = threshold_idx(
        &ma(&cc_mean_through_mask, 4),
        mc_thr
    );
    let cc_d_steps_threshold = sum_left_slice(
        &cc_mean_through_mask,
        cc_d_threshold_idx
    );
    cc_mean_through_mask.truncate(cc_d_threshold_idx);

    params.XC_D_STEPS_COUNT = params.NODES
        - lc_d_steps_threshold as usize
        - qc_d_steps_threshold as usize
        - cc_d_steps_threshold as usize;

    println!("\tlc_d_steps_threshold: {}", lc_d_steps_threshold);
    println!("\tqc_d_steps_threshold: {}", qc_d_steps_threshold);
    println!("\tcc_d_steps_threshold: {}", cc_d_steps_threshold);
    println!("\tXC_D_STEPS_COUNT set to {}", params.XC_D_STEPS_COUNT);

    let mut xc_mean_through_mask: Vec<f64> = perform_testing_xc(&node_matrix, &input_matrix, rng, params);

    let xc_d_threshold_idx = threshold_idx(
        &ma(&xc_mean_through_mask, 4),
        mc_thr
    );
    xc_mean_through_mask.truncate(xc_d_threshold_idx);

    println!("\n\tLC len: {}\n\tQC len: {}\n\tXC len: {}\n", lc_mean_through_mask.len(), qc_mean_through_mask.len(), xc_mean_through_mask.len());

    return (lc_mean_through_mask, qc_mean_through_mask, cc_mean_through_mask, xc_mean_through_mask);
}


pub fn perform_testing_lc_qc(node_matrix: &DMatrix<f64>, input_matrix: &DMatrix<f64>, rng: &mut StdRng, params: &params::Params) -> (Vec<f64>, Vec<f64>) {
    let mut lcs_through_folds: Vec<Vec<f64>> = Vec::with_capacity(constants::MC_K_FOLDS_COUNT);
    let mut qcs_through_folds: Vec<Vec<f64>> = Vec::with_capacity(constants::MC_K_FOLDS_COUNT);

    for k in 0..constants::MC_K_FOLDS_COUNT {
        if k % constants::MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL as usize == 0 {
            println!("LC QC Validation... {:.1} %", k as f64 / constants::MC_K_FOLDS_COUNT as f64 * 100.);
        }

        // usable range of indices: (total_len - warm_up)
        let mut values: Vec<usize> = (0..(constants::SAMPLES_COUNT - constants::WARM_UP_COUNT)).collect();
        values.shuffle(rng);
        // take test_set_len=200 random indices
        let test_idxs: Vec<usize> = values.clone().into_iter().take(constants::MC_TEST_SET_LEN).collect();
        
        // make hash set of test indices
        let selected: HashSet<usize> = test_idxs.iter().cloned().collect();
        // take all of usable range of indices but test indices
        let train_idxs: Vec<usize> = values.clone().into_iter().filter(|x| !selected.contains(x)).collect();
        
        let mut lc_input_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 200x500
        let mut lc_input_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 1800x500

        let mut qc_input_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 200x500
        let mut qc_input_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 1800x500
        
        // tmp vectors
        let mut lc_tmp_test: Vec<f64> = Vec::with_capacity(constants::MC_TEST_SET_LEN);
        let mut lc_tmp_train: Vec<f64> = Vec::with_capacity(constants::MC_TRAIN_SET_LEN);
        
        let mut qc_tmp_test: Vec<f64> = Vec::with_capacity(constants::MC_TEST_SET_LEN);
        let mut qc_tmp_train: Vec<f64> = Vec::with_capacity(constants::MC_TRAIN_SET_LEN);

        for j in 1..params.D_STEPS_COUNT {
            lc_tmp_test.clear();
            lc_tmp_train.clear();
            qc_tmp_test.clear();
            qc_tmp_train.clear();

            // fill tmp test vector with values of input_matrix from row=(one of test indices + warm_up - j)
            for row_idx in &test_idxs {
                let tmp_get = input_matrix[(row_idx+constants::WARM_UP_COUNT-j, 0)];
                lc_tmp_test.push(shift_range(tmp_get, -0.5, 2.));
                qc_tmp_test.push(3. * shift_range(tmp_get, -0.5, 2.).powi(2) - 1.);
            }
            // fill tmp train vector with values of input_matrix from row=(one of train indices + warm_up - j)
            for row_idx in &train_idxs {
                let tmp_get = input_matrix[(row_idx+constants::WARM_UP_COUNT-j, 0)];
                lc_tmp_train.push(shift_range(tmp_get, -0.5, 2.));
                qc_tmp_train.push(3. * shift_range(tmp_get, -0.5, 2.).powi(2) - 1.);
            }

            // set (j-1)-column of matrixes from tmp 'delayed' vectors
            lc_input_test.column_mut(j-1).copy_from_slice(&lc_tmp_test);
            lc_input_train.column_mut(j-1).copy_from_slice(&lc_tmp_train);
            qc_input_test.column_mut(j-1).copy_from_slice(&qc_tmp_test);
            qc_input_train.column_mut(j-1).copy_from_slice(&qc_tmp_train);
        }

        let mut nodes_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.NODES+1
        ); // 200x10+1
        let mut nodes_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.NODES+1
        ); // 1800x10+1

        // fill node states matrixes without 'delay'
        for i in 0..test_idxs.len() {
            nodes_test.row_mut(i).copy_from(&node_matrix.row(test_idxs[i]+constants::WARM_UP_COUNT));
        }
        for i in 0..train_idxs.len() {
            nodes_train.row_mut(i).copy_from(&node_matrix.row(train_idxs[i]+constants::WARM_UP_COUNT));
        }

        // obtain readout weights
        let lc_w =
            (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
            * &nodes_train.transpose() * &lc_input_train;

        let qc_w =
            (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
            * &nodes_train.transpose() * &qc_input_train;

        // test_set_len x max_d_steps (200x500)
        let lc_predicted = &nodes_test * &lc_w;
        let qc_predicted = &nodes_test * &qc_w;

        let mut lc_ds: Vec<f64> = Vec::with_capacity(params.D_STEPS_COUNT-1);
        let mut qc_ds: Vec<f64> = Vec::with_capacity(params.D_STEPS_COUNT-1);

        for j in 1..params.D_STEPS_COUNT {
            // transform j-column of predicted matrix of size 200x500 to Vec
            let lc_tmp_vec_predicted: Vec<f64> = lc_predicted.column(j-1).iter().copied().collect();
            // transform j-column of input_test matrix of size 200x500 to Vec
            let lc_tmp_vec_input: Vec<f64> = lc_input_test.column(j-1).iter().copied().collect();

            let qc_tmp_vec_predicted: Vec<f64> = qc_predicted.column(j-1).iter().copied().collect();
            let qc_tmp_vec_input: Vec<f64> = qc_input_test.column(j-1).iter().copied().collect();

            // calculate correlation as cov^2/(stdev * stdev)
            // lc_ds.push(
            //     utils::pearson_squared(&lc_tmp_vec_predicted, &lc_tmp_vec_input)
            // );
            // qc_ds.push(
            //     utils::pearson_squared(&qc_tmp_vec_predicted, &qc_tmp_vec_input)
            // );

            lc_ds.push(
                1. - nmse(&lc_tmp_vec_input, &lc_tmp_vec_predicted)
            );
            qc_ds.push(
                1. - nmse(&qc_tmp_vec_input, &qc_tmp_vec_predicted)
            );
        }

        lcs_through_folds.push(lc_ds);
        qcs_through_folds.push(qc_ds);
    }

    let mut lc_mean_through_mask: Vec<f64> = vec![0.; params.D_STEPS_COUNT-1];
    let mut qc_mean_through_mask: Vec<f64> = vec![0.; params.D_STEPS_COUNT-1];

    for i in 0..constants::MC_K_FOLDS_COUNT {
        for j in 0..(params.D_STEPS_COUNT-1) {
            lc_mean_through_mask[j] += lcs_through_folds[i][j];
            qc_mean_through_mask[j] += qcs_through_folds[i][j];
        }
    }

    for j in 0..(params.D_STEPS_COUNT-1) {
        lc_mean_through_mask[j] /= constants::MC_K_FOLDS_COUNT as f64;
        qc_mean_through_mask[j] /= constants::MC_K_FOLDS_COUNT as f64;
    }

    return (lc_mean_through_mask, qc_mean_through_mask);
}


pub fn perform_testing_cc(node_matrix: &DMatrix<f64>, input_matrix: &DMatrix<f64>, rng: &mut StdRng, params: &params::Params) -> Vec<f64> {
    let mut ccs_through_folds: Vec<Vec<f64>> = Vec::with_capacity(constants::MC_K_FOLDS_COUNT);

    for k in 0..constants::MC_K_FOLDS_COUNT {
        if k % constants::MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL as usize == 0 {
            println!("CC Validation... {:.1} %", k as f64 / constants::MC_K_FOLDS_COUNT as f64 * 100.);
        }

        // usable range of indices: (total_len - warm_up)
        let mut values: Vec<usize> = (0..(constants::SAMPLES_COUNT - constants::WARM_UP_COUNT)).collect();
        values.shuffle(rng);
        // take test_set_len=200 random indices
        let test_idxs: Vec<usize> = values.clone().into_iter().take(constants::MC_TEST_SET_LEN).collect();
        
        // make hash set of test indices
        let selected: HashSet<usize> = test_idxs.iter().cloned().collect();
        // take all of usable range of indices but test indices
        let train_idxs: Vec<usize> = values.clone().into_iter().filter(|x| !selected.contains(x)).collect();
        
        let mut cc_input_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 200x500
        let mut cc_input_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.D_STEPS_COUNT-1
        ); // 1800x500
        
        // tmp vectors
        let mut cc_tmp_test: Vec<f64> = Vec::with_capacity(constants::MC_TEST_SET_LEN);
        let mut cc_tmp_train: Vec<f64> = Vec::with_capacity(constants::MC_TRAIN_SET_LEN);

        for j in 1..params.D_STEPS_COUNT {
            cc_tmp_test.clear();
            cc_tmp_train.clear();

            // fill tmp test vector with values of input_matrix from row=(one of test indices + warm_up - j)
            for row_idx in &test_idxs {
                let tmp_get = input_matrix[(row_idx+constants::WARM_UP_COUNT-j, 0)];
                cc_tmp_test.push(leg_poly_3_order(shift_range(tmp_get, -0.5, 2.)));
            }
            // fill tmp train vector with values of input_matrix from row=(one of train indices + warm_up - j)
            for row_idx in &train_idxs {
                let tmp_get = input_matrix[(row_idx+constants::WARM_UP_COUNT-j, 0)];
                cc_tmp_train.push(leg_poly_3_order(shift_range(tmp_get, -0.5, 2.)));
            }

            // set (j-1)-column of matrixes from tmp 'delayed' vectors
            cc_input_test.column_mut(j-1).copy_from_slice(&cc_tmp_test);
            cc_input_train.column_mut(j-1).copy_from_slice(&cc_tmp_train);
        }

        let mut nodes_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.NODES+1
        ); // 200x10+1
        let mut nodes_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.NODES+1
        ); // 1800x10+1

        // fill node states matrixes without 'delay'
        for i in 0..test_idxs.len() {
            nodes_test.row_mut(i).copy_from(&node_matrix.row(test_idxs[i]+constants::WARM_UP_COUNT));
        }
        for i in 0..train_idxs.len() {
            nodes_train.row_mut(i).copy_from(&node_matrix.row(train_idxs[i]+constants::WARM_UP_COUNT));
        }

        // obtain readout weights
        let cc_w =
            (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
            * &nodes_train.transpose() * &cc_input_train;

        // test_set_len x max_d_steps (200x500)
        let cc_predicted = &nodes_test * &cc_w;

        let mut cc_ds: Vec<f64> = Vec::with_capacity(params.D_STEPS_COUNT-1);

        for j in 1..params.D_STEPS_COUNT {
            // transform j-column of predicted matrix of size 200x500 to Vec
            let cc_tmp_vec_predicted: Vec<f64> = cc_predicted.column(j-1).iter().copied().collect();
            // transform j-column of input_test matrix of size 200x500 to Vec
            let cc_tmp_vec_input: Vec<f64> = cc_input_test.column(j-1).iter().copied().collect();

            // calculate correlation as cov^2/(stdev * stdev)
            cc_ds.push(
                1. - nmse(&cc_tmp_vec_input, &cc_tmp_vec_predicted)
            );
        }

        ccs_through_folds.push(cc_ds);
    }

    let mut cc_mean_through_mask: Vec<f64> = vec![0.; params.D_STEPS_COUNT-1];

    for i in 0..constants::MC_K_FOLDS_COUNT {
        for j in 0..(params.D_STEPS_COUNT-1) {
            cc_mean_through_mask[j] += ccs_through_folds[i][j];
        }
    }

    for j in 0..(params.D_STEPS_COUNT-1) {
        cc_mean_through_mask[j] /= constants::MC_K_FOLDS_COUNT as f64;
    }

    return cc_mean_through_mask;
}


pub fn perform_testing_xc(node_matrix: &DMatrix<f64>, input_matrix: &DMatrix<f64>, rng: &mut StdRng, params: &params::Params) -> Vec<f64> {
    let mut xcs_through_folds: Vec<Vec<f64>> = Vec::with_capacity(constants::MC_K_FOLDS_COUNT);

    let l_max: usize = params.XC_D_STEPS_COUNT;
    let num_pairs: usize = (l_max * (l_max - 1)) / 2;

    println!("\tnum_pairs (XC) = {}", num_pairs);

    for k in 0..constants::MC_K_FOLDS_COUNT {
        if k % constants::MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL as usize == 0 {
            println!("XC Validation... {:.1} %", k as f64 / constants::MC_K_FOLDS_COUNT as f64 * 100.);
        }

        // usable range of indices: (total_len - warm_up)
        let mut values: Vec<usize> = (0..(constants::SAMPLES_COUNT - constants::WARM_UP_COUNT)).collect();
        values.shuffle(rng);

        // take test_set_len=200 random indices
        let test_idxs: Vec<usize> = values.clone().into_iter().take(constants::MC_TEST_SET_LEN).collect();
        
        // make hash set of test indices
        let selected: HashSet<usize> = test_idxs.iter().cloned().collect();
        // take all of usable range of indices but test indices
        let train_idxs: Vec<usize> = values.clone().into_iter().filter(|x| !selected.contains(x)).collect();

        // COMPUTING XC

        let mut xc_input_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            num_pairs
        );
        let mut xc_input_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            num_pairs
        );

        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(num_pairs);
        for l in 0..l_max {
            for l_prime in (l + 1)..l_max {
                pairs.push((l, l_prime));
            }
        }

        // making XC train set
        let mut tmp_vec = Vec::with_capacity(constants::MC_TRAIN_SET_LEN);
        for (col_idx, &(l, l_prime)) in pairs.iter().enumerate() {
            for row_idx in &train_idxs {
                let val_l = input_matrix[(row_idx + constants::WARM_UP_COUNT - l, 0)];
                let val_l_prime = input_matrix[(row_idx + constants::WARM_UP_COUNT - l_prime, 0)];
                tmp_vec.push(
                    shift_range(val_l, -0.5, 2.) * shift_range(val_l_prime, -0.5, 2.)
                );
            }
            xc_input_train.column_mut(col_idx).copy_from_slice(&tmp_vec);
            tmp_vec.clear();
        }

        // making XC test set
        let mut tmp_vec = Vec::with_capacity(constants::MC_TEST_SET_LEN);
        for (col_idx, &(l, l_prime)) in pairs.iter().enumerate() {
            for row_idx in &test_idxs {
                let val_l = input_matrix[(row_idx + constants::WARM_UP_COUNT - l, 0)];
                let val_l_prime = input_matrix[(row_idx + constants::WARM_UP_COUNT - l_prime, 0)];
                tmp_vec.push(
                    shift_range(val_l, -0.5, 2.) * shift_range(val_l_prime, -0.5, 2.)
                );
            }
            xc_input_test.column_mut(col_idx).copy_from_slice(&tmp_vec);
            tmp_vec.clear();
        }

        let mut nodes_test = DMatrix::<f64>::zeros(
            constants::MC_TEST_SET_LEN,
            params.NODES+1
        ); // 200x10+1
        let mut nodes_train = DMatrix::<f64>::zeros(
            constants::MC_TRAIN_SET_LEN,
            params.NODES+1
        ); // 1800x10+1

        // fill node states matrixes without 'delay'
        for i in 0..test_idxs.len() {
            nodes_test.row_mut(i).copy_from(&node_matrix.row(test_idxs[i]+constants::WARM_UP_COUNT));
        }
        for i in 0..train_idxs.len() {
            nodes_train.row_mut(i).copy_from(&node_matrix.row(train_idxs[i]+constants::WARM_UP_COUNT));
        }

        // learning
        let xc_w =
            (nodes_train.transpose() * &nodes_train).try_inverse().unwrap()
            * &nodes_train.transpose() * &xc_input_train;

        let xc_predicted = &nodes_test * &xc_w;

        let mut xc_ds: Vec<f64> = Vec::with_capacity(num_pairs);

        for col_idx in 0..num_pairs {
            let xc_tmp_predicted = xc_predicted.column(col_idx).iter().copied().collect();
            let xc_tmp_input = xc_input_test.column(col_idx).iter().copied().collect();

            // xc_ds.push(
            //     utils::pearson_squared(&xc_tmp_predicted, &xc_tmp_input)
            // );

            xc_ds.push(
                1. - nmse(&xc_tmp_input, &xc_tmp_predicted)
            );
        }

        xcs_through_folds.push(xc_ds);
    }

    let mut xc_mean_through_mask: Vec<f64> = vec![0.; num_pairs];

    for i in 0..constants::MC_K_FOLDS_COUNT {
        for j in 0..num_pairs {
            xc_mean_through_mask[j] += xcs_through_folds[i][j];
        }
    }

    for j in 0..num_pairs {
        xc_mean_through_mask[j] /= constants::MC_K_FOLDS_COUNT as f64;
    }

    return xc_mean_through_mask;
}


pub fn shift_range(x: f64, add: f64, mult: f64) -> f64 {
    return (x + add) * mult;
}

pub fn leg_poly_3_order(x: f64) -> f64 {
    return 0.5 * (5. * x.powi(3) - 3. * x);
}

pub fn sum_left_slice(data: &Vec<f64>, until: usize) -> f64 {
    let mut sum = 0.;
    for i in 0..until {
        sum += data[i];
    }

    return sum;
}

pub fn nmse(y_true: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    let mse = y_true.iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / y_true.len() as f64;

    return mse / stat::variance(y_true);
}

pub fn threshold_idx(data: &Vec<f64>, threshold_less_than: f64) -> usize {
    for i in 0..data.len() {
        if data[i] < threshold_less_than {
            return i;
        }
    }

    return data.len();
}

pub fn ma(data: &Vec<f64>, window: usize) -> Vec<f64> {
    if data.len() < window {
        println!("WARNING utils.rs: MA: window len is less than data len!");
        return data.clone();
    }

    let mut res: Vec<f64> = Vec::with_capacity(data.len() - window);
    for i in window..data.len() {
        let mut sum = 0.;
        for j in (i-window)..i {
            sum += data[j];
        }
        res.push(sum / window as f64);
    }

    return res;
}