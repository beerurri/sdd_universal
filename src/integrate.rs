use rand_distr::Distribution;

use crate::{aliases, params};

pub fn integrate<'a>(
    f: aliases::F<'a>,
    y0: f64,
    t_stop: f64,
    step_callback: aliases::C<'a>,
    output_nodes: &mut Vec<f64>,
    params: &mut params::Params,
    input_data: &Vec<f64>,
    mask: &Vec<f64>
) {
    let mut t_curr = 0.0;
    // let mut y_new = [0.0; constants::PROBLEM_DIM];
    let mut counter: usize = 0;

    let mut hist: Vec<f64> = Vec::with_capacity(params.MAX_HIST_LEN);

    let nodes_period = (params.Theta / params.DT) as usize;
    let nodes_halfperiod = nodes_period / 2;

    hist.push(y0);


    while t_curr < t_stop {
        let hist_len = hist.len();
        let y_curr = &hist[hist_len - 1];
        let y_new = em_step(f, *y_curr, t_curr, &hist, params, input_data, mask);

        if hist_len == params.MAX_HIST_LEN {
            hist.drain(0..(params.MAX_HIST_LEN / 2));
        }
        
        hist.push(y_new);

        if (counter + 1) >= nodes_halfperiod && ((counter + 1) - nodes_halfperiod) % nodes_period == 0 {
            output_nodes.push(y_new);
        }

        step_callback(counter, &hist, params);

        counter += 1;
        t_curr += params.DT;
    }

    if output_nodes.len() > params.TOTAL_OUTPUT_NODES {
        output_nodes.drain(params.TOTAL_OUTPUT_NODES..output_nodes.len());
    }
}

/// RK4 step function
/// 
/// `f`: right-hand side function
/// 
/// `y_curr`: current `y`
/// 
/// `t_curr`: current time
/// 
/// `dt`: time step
/// 
/// `get_hist`: function that returns historical `y` value
// pub fn step<'a>(f: aliases::F<'a>, y_curr: f64, t_curr: f64, hist: &Vec<f64>, params: &mut params::Params, input_data: &Vec<f64>, mask: &Vec<f64>, counter: u64, recorder: &mut utils::Recorder) -> f64 {
//     let dt_div_2 = constants::DT / 2.0;
    
//     let k1 = [f(t_curr, y_curr, hist, params, input_data, mask, counter, recorder)[0] + noise_sample];
    
//     let mut tmp_1 = [0.0; constants::PROBLEM_DIM];
//     let mut tmp_2 = [0.0; constants::PROBLEM_DIM];

//     vector_utils::vec_scalar_mult(&k1, dt_div_2, &mut tmp_1);
//     vector_utils::vec_summ(y_curr, &tmp_1, &mut tmp_2);
    
//     let k2 = f(
//         t_curr + dt_div_2, &tmp_2, hist, params, input_data, mask, counter, recorder
//     );

//     vector_utils::vec_scalar_mult(&k2, dt_div_2, &mut tmp_1);
//     vector_utils::vec_summ(y_curr, &tmp_1, &mut tmp_2);

//     let k3 = f(
//         t_curr + dt_div_2, &tmp_2, hist, params, input_data, mask, counter, recorder
//     );

//     vector_utils::vec_scalar_mult(&k3, constants::DT, &mut tmp_1);
//     vector_utils::vec_summ(y_curr, &tmp_1, &mut tmp_2);

//     let k4 = f(
//         t_curr + constants::DT, &tmp_2, hist, params, input_data, mask, counter, recorder
//     );

//     // calculating dy = (h/6)*(k1 + 2*k2 + 2*k3 + k4) = (h/6)*(k1 + 2*(k2 + k3) + k4)
//     vector_utils::vec_summ(&k2, &k3, &mut tmp_1); // k2 + k3
//     vector_utils::vec_scalar_mult(&tmp_1, 2.0, &mut tmp_2); // 2*(k2 + k3)
//     vector_utils::vec_summ(&k1, &tmp_2, &mut tmp_1); // k1 + 2*(k2 + k3)
//     vector_utils::vec_summ(&tmp_1, &k4, &mut tmp_2); // k1 + 2*(k2 + k3) + k4

//     vector_utils::vec_scalar_mult(&tmp_2, constants::DT / 6.0, &mut tmp_1); // dy = (h/6)*(k1 + 2*(k2 + k3) + k4)

//     let mut y_new = [0.0; constants::PROBLEM_DIM];

//     vector_utils::vec_summ(y_curr, &tmp_1, &mut y_new);

//     return y_new;
// }

pub fn em_step<'a>(f: aliases::F<'a>, y_curr: f64, t_curr: f64, hist: &Vec<f64>, params: &mut params::Params, input_data: &Vec<f64>, mask: &Vec<f64>) -> f64 {
    let noise_sample = params.normal_distr.sample(&mut params.noise_rng);
    
    let y_new =
        y_curr +
        f(t_curr + params.DT, y_curr, hist, params, input_data, mask) * params.DT +
        noise_sample * params.DT.sqrt();

    return y_new;
}