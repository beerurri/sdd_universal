mod mc;
mod constants;
mod utils;
mod params;
mod aliases;
mod integrate;
mod santa;
mod narma;

use clap::Parser;
use stopwatch::Stopwatch;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io;

#[derive(Parser)]
#[command(name = "universal")]
struct Args {
    /// Task to perform
    #[arg(long)]
    task: String,

    /// Case name for file saving
    #[arg(long, default_value = "default")]
    case: String,

    /// Theta parameter
    #[arg(long, default_value = "0.12")]
    theta: f64,

    /// Beta parameter
    #[arg(long, default_value = "0.5")]
    beta: f64,

    /// Tau_0 parameter
    #[arg(long, default_value = "1.0")]
    tau_0: f64,

    /// Every second flag
    #[arg(long, default_value = "false")]
    every_second: bool,

    /// is Theta constant
    #[arg(long, default_value = "false")]
    constant_theta: bool
}

fn main() {
    let args = Args::parse();

    if args.every_second && args.constant_theta {
        panic!("Error: using every second Node while constant Theta is forbidden");
    }

    let mut rng = StdRng::seed_from_u64(2025);
    let total_sw = Stopwatch::new();

    let mut params = params::Params::defaults();
    params.Theta = args.theta;
    params.beta = args.beta;
    params.tau_0 = args.tau_0;
    params.every_second = args.every_second;
    params.update(&args);

    println!("Task: {:?},\n\tTheta: {},\n\tTau_0: {},\n\tNODES: {},\n\tDT: {},\n\tBeta: {},\n\tEvery Second: {},\n\tConstant Theta: {}\n\tCase: {},\n\tTRAINING_NODES: {}",
             args.task, params.Theta, params.tau_0, params.NODES, params.DT, params.beta, params.every_second, args.constant_theta, args.case, params.TRAINING_NODES);

    match args.task.as_str() {
        "santa" => {
            run_santa(&mut params, &mut rng, &args);
        },
        "narma" => {
            run_narma(&mut params, &mut rng, &args);
        },
        "mc" => {
            run_mc(&mut params, &mut rng, &args);
        },
        _ => println!("Unknown task: {}", args.task)
    }

    let total_sw_stop = total_sw.elapsed();
    println!("Total elapsed {} ms", total_sw_stop.as_nanos() as f64 / 1e6);
}


fn run_santa(params: &mut params::Params, rng: &mut StdRng, args: &Args) {
    let (input_data, output_data) = santa::make_santa_data();

    let mut nmses: Vec<f64> = Vec::new();

    for mask_idx in 0..constants::MASKS_COUNT {
        println!("\n\n\tMask index: {}\n", mask_idx);
        let mask = utils::generate_mask(rng, &params);

        let mut node_states: Vec<f64> = Vec::with_capacity(params.TOTAL_OUTPUT_NODES);

        let f_ref: &dyn Fn(f64, f64, &Vec<f64>, &mut params::Params, &Vec<f64>, &Vec<f64>) -> f64 = &rhs;
        let step_callback_ref: &dyn Fn(usize, &Vec<f64>, &params::Params) -> () = &step_callback;

        let mut sw = Stopwatch::start_new();
        integrate::integrate(f_ref, params.Y0, params.TOTAL_TIME_TO_COMPUTE, step_callback_ref, &mut node_states, params, &input_data, &mask);

        let mut stop = sw.elapsed();
        println!("Calculation elapsed {} ms", stop.as_nanos() as f64 / 1e6);

        println!("Testing...");
        sw = Stopwatch::start_new();
        let mean_nmse: f64 = santa::perform_testing(&node_states, &output_data, &params);
        nmses.push(mean_nmse);
        println!("Mean NMSE at mask #{}: {}", mask_idx, mean_nmse);
        stop = sw.elapsed();
        println!("Testing elapsed {} ms", stop.as_nanos() as f64 / 1e6);

        if constants::LOG_NODES {
            println!("Saving nodes log...");
            _ = utils::save_f64_vector_to_csv(&node_states[(node_states.len()-constants::LOG_NODES_COUNT)..node_states.len()].to_vec(), "nodes_log.csv");
            
            println!("Continue?");

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            if input.trim().is_empty() {
                println!("Continuing...");
            } else {
                println!("Exiting...");
                std::process::exit(0);
            }
        }

        if constants::LOG_ALL_NODES_BY_MASK {
            println!("Saving nodes log, mask {}...", mask_idx);
            _ = utils::save_f64_vector_to_csv(&node_states, &format!("node_logs/nodes_log_mask-{mask_idx}.csv"))
        }
    }

    if constants::LOG_NODES {
        std::process::exit(0);
    }

    println!("Mean NMSE through {} masks: {}", constants::MASKS_COUNT, nmses.iter().sum::<f64>() / nmses.len() as f64);

    println!("Saving...");
    _ = utils::save_f64_vector_to_csv(&nmses, &utils::filename_builder(args, &params, ""));
}


fn run_narma(params: &mut params::Params, rng: &mut StdRng, args: &Args) {
    let (input_data, output_data) = narma::make_narma10_data(rng, constants::SAMPLES_COUNT);

    let mut nrmses: Vec<f64> = Vec::new();

    for mask_idx in 0..constants::MASKS_COUNT {
        println!("\n\n\tMask index: {}\n", mask_idx);
        let mask = utils::generate_mask(rng, &params);

        let mut node_states: Vec<f64> = Vec::with_capacity(params.TOTAL_OUTPUT_NODES);

        let f_ref: &dyn Fn(f64, f64, &Vec<f64>, &mut params::Params, &Vec<f64>, &Vec<f64>) -> f64 = &rhs;
        let step_callback_ref: &dyn Fn(usize, &Vec<f64>, &params::Params) -> () = &step_callback;

        let mut sw = Stopwatch::start_new();
        integrate::integrate(f_ref, params.Y0, params.TOTAL_TIME_TO_COMPUTE, step_callback_ref, &mut node_states, params, &input_data, &mask);

        let mut stop = sw.elapsed();
        println!("Calculation elapsed {} ms", stop.as_nanos() as f64 / 1e6);

        println!("Testing...");
        sw = Stopwatch::start_new();
        let nrmse: f64 = narma::perform_testing(&node_states, &output_data, &params);
        nrmses.push(nrmse);
        println!("Mean NRMSE at mask #{}: {}", mask_idx, nrmse);
        stop = sw.elapsed();
        println!("Testing elapsed {} ms", stop.as_nanos() as f64 / 1e6);
    }

    println!("Mean NMSE through {} masks: {}", constants::MASKS_COUNT, nrmses.iter().sum::<f64>() / nrmses.len() as f64);

    println!("Saving...");
    _ = utils::save_f64_vector_to_csv(&nrmses, &utils::filename_builder(args, &params, ""));
}


fn run_mc(params: &mut params::Params, rng: &mut StdRng, args: &Args) {
    let input_data = utils::generate_input_uniform(rng);

    let mut lcs: Vec<Vec<f64>> = Vec::with_capacity(constants::MASKS_COUNT);
    let mut qcs: Vec<Vec<f64>> = Vec::with_capacity(constants::MASKS_COUNT);
    let mut ccs: Vec<Vec<f64>> = Vec::with_capacity(constants::MASKS_COUNT);
    let mut xcs: Vec<Vec<f64>> = Vec::with_capacity(constants::MASKS_COUNT);

    for mask_idx in 0..constants::MASKS_COUNT {
        println!("\n\n\tMask index: {}\n", mask_idx);
        let mask = utils::generate_mask(rng, &params);

        let mut node_states: Vec<f64> = Vec::with_capacity(params.TOTAL_OUTPUT_NODES);

        let f_ref: aliases::F = &rhs;
        let step_callback_ref: aliases::C = &step_callback;

        let mut sw = Stopwatch::start_new();

        integrate::integrate(f_ref, params.Y0, params.TOTAL_TIME_TO_COMPUTE, step_callback_ref, &mut node_states, params, &input_data, &mask);
        
        let mut stop = sw.elapsed();
        println!("Calculation elapsed {} ms", stop.as_nanos() as f64 / 1e6);

        println!("Testing...");
        sw = Stopwatch::start_new();
        let (lc_per_mask, qc_per_mask, cc_per_mask, xc_per_mask) = mc::perform_testing(&input_data, &node_states, rng, params);
        stop = sw.elapsed();
        lcs.push(lc_per_mask);
        qcs.push(qc_per_mask);
        ccs.push(cc_per_mask);
        xcs.push(xc_per_mask);
        println!("Testing elapsed {} ms", stop.as_nanos() as f64 / 1e6);
    }

    println!("Saving...");
    _ = utils::save_to_csv(&lcs, &utils::filename_builder(args, &params, "LC"));
    _ = utils::save_to_csv(&qcs, &utils::filename_builder(args, &params, "QC"));
    _ = utils::save_to_csv(&ccs, &utils::filename_builder(args, &params, "CC"));
    _ = utils::save_to_csv(&xcs, &utils::filename_builder(args, &params, "XC"));
}


fn rhs(curr_t: f64, curr_y: f64, hist: &Vec<f64>, params: &mut params::Params, input_data: &Vec<f64>, mask: &Vec<f64>) -> f64 {
    return (
            -curr_y -
            params.k * get_hist(curr_y, hist, params) + 
            params.alpha * modulation_f(curr_t, input_data, mask, params)
        ) / params.eps;
}

fn get_hist(y_curr: f64, hist: &Vec<f64>, params: &params::Params) -> f64 {
    // let t_past = t_curr - (constants::TAU_0 + params.beta * y_curr[constants::GET_HIST_EQUATION_IDX]);
    // let idx_past = (t_past / constants::DT) as isize;

    let tau = params.tau_0 + params.beta * y_curr;
    let idx_past = hist.len() as isize - (tau / params.DT) as isize;

    if idx_past <= 0 {
        return params.Y0;
    } else if idx_past >= hist.len() as isize {
        return y_curr;
    } else {
        return hist[idx_past as usize];
    }
}

fn modulation_f(curr_t: f64, input_data: &Vec<f64>, mask: &Vec<f64>, params: &params::Params) -> f64 {
    let mut idx_t_cc = (curr_t / params.T_cc) as usize;
    let idx_theta = (curr_t / params.Theta % params.NODES as f64) as usize;

    if idx_t_cc >= constants::SAMPLES_COUNT {
        idx_t_cc = constants::SAMPLES_COUNT - 1;
    }

    let modulation = input_data[idx_t_cc] * mask[idx_theta];

    return modulation;
}

fn step_callback(counter: usize, hist: &Vec<f64>, params: &params::Params) {
    // if counter == 60000 {
    //     std::process::exit(0);
    // }
    if counter % params.CALLBACK_CONSOLE_OUTPUT_INTERVAL == 0 {
        // println!("[\n\t{},\t{},\t{},\t{}\n]", hist[counter as usize][0], hist[counter as usize][1], hist[counter as usize][2],hist[counter as usize][3]);
        println!("{:.1} %", (counter as f64 / params.TOTAL_ITERS_TO_COMPUTE as f64 * 100.));
        // println!("{:.6}, {:.6}", hist[hist.len()-1][0], hist[hist.len()-1][1]);
    }
}