use std::error::Error;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::SystemTime;
use csv::Writer;
use crate::constants;
use rand::rngs::StdRng;
use rand::Rng;
use rand::distr::Uniform;
use crate::params;
// use flate2::{write::ZlibEncoder, read::ZlibDecoder, Compression};
// use bincode::{config, serde::{encode_to_vec, decode_from_slice}};
// use serde::{Serialize, Deserialize};
// use blosc::{Clevel, Context};


// pub fn free_2d_vec_u8(mut data: Vec<Vec<u8>>) {
//     for v in &mut data {
//         v.clear();
//         v.shrink_to_fit();
//     }
//     data.clear();
//     data.shrink_to_fit();
//     drop(data);
// }

// pub fn free_2d_vec_f64(mut data: Vec<Vec<f64>>) {
//     for v in &mut data {
//         v.clear();
//         v.shrink_to_fit();
//     }
//     data.clear();
//     data.shrink_to_fit();
//     drop(data);
// }

// pub fn compress_vec_f64(data: &Vec<f64>) -> Vec<u8> {
//     // let encoded = encode_to_vec(&data, config::standard()).unwrap();

//     // let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
//     // encoder.write_all(&encoded).unwrap();
//     // return encoder.finish().unwrap();

//     let ctx = Context::new().clevel(Clevel::L9);
//     return ctx.compress(&data).into();
// }

// pub fn decompress2vec_f64(compressed: &[u8]) -> Vec<f64> {
//     // let mut decoder = ZlibDecoder::new(compressed);
//     // let mut decoded = Vec::new();
//     // decoder.read_to_end(&mut decoded).unwrap();

//     // return decode_from_slice(&decoded, config::standard()).unwrap().0;

//     return unsafe {
//         blosc::decompress_bytes(compressed).unwrap()
//     };
// }

pub fn filename_builder(args: &crate::Args, params: &params::Params, extra: &str) -> String {
    let mut res = format!(
        "results/{}_case_{}_theta_{}_beta_{}_tau0_{}_every_second_{}_constant-theta_{}",
        args.task, args.case, params.Theta, params.beta, params.tau_0, args.every_second, args.constant_theta
    );

    if extra.len() > 0 {
        res.push_str(&format!("_{}.csv", extra));
    } else {
        res.push_str(".csv");
    }

    return res;
}

pub fn save_to_csv(data: &Vec<Vec<f64>>, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for (i, row) in data.iter().enumerate() {
        let line = row.iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join(",");

        if i + 1 < data.len() {
            writeln!(writer, "{}", line)?;
        } else {
            write!(writer, "{}", line)?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub fn save_vector_to_csv(data: &Vec<f64>, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let mut wtr = Writer::from_writer(file);

    wtr.serialize(data)?;

    wtr.flush()?;
    Ok(())
}

// pub fn make_santa_data() -> (Vec<f64>, Vec<f64>) {
//     let file = File::open("input_data.dat").unwrap();
//     let reader = BufReader::new(file);

//     let mut numbers: Vec<f64> = reader
//         .lines()
//         .filter_map(|line| line.unwrap().trim().parse::<f64>().ok())
//         .collect();

//     let min= numbers.iter().cloned().reduce(f64::min).unwrap();
//     let max = numbers.iter().cloned().reduce(f64::max).unwrap();

//     numbers = numbers.iter().map(|&x| (x - min) / max).collect();

//     return (numbers[0..constants::TOTAL_INPUT_FEATURES].to_vec(), numbers[constants::SANTA_K_STEPS..(constants::TOTAL_OUTPUT_FEATURES+constants::SANTA_K_STEPS)].to_vec());
// }

pub fn generate_mask(rng: &mut StdRng, params: &params::Params) -> Vec<f64> {
    return (0..params.NODES as usize)
        .map(|_| rng.random_range(-1.0..=1.0))
        .collect();
}

pub fn generate_input_uniform(rng: &mut StdRng) -> Vec<f64> {
    let uniform = Uniform::new_inclusive(0., 1.).unwrap();
    return (0..constants::SAMPLES_COUNT)
        .map(|_| rng.sample(uniform))
        .collect();
}

pub fn pearson_squared(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let n = x.len();

    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // unbiased (n - 1) or biased (n)
    cov /= (n - 1) as f64;
    var_x /= (n - 1) as f64;
    var_y /= (n - 1) as f64;

    (cov * cov) / (var_x * var_y)
}

pub fn log(filename: &str, params: &params::Params, string: &str) {
    println!("\n\tLogging...\n");

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename).unwrap();

    let now = SystemTime::now();
    let timestamp = now.duration_since(std::time::UNIX_EPOCH).unwrap();

    let mut string_params = timestamp.as_millis().to_string() + ",";
    // string_params.push_str(&format!(
    //     "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
    //     0,
    //     0,
    //     0,
    //     constants::NODES,
    //     constants::T_cc,
    //     params.tau_0,
    //     constants::Theta,
    //     constants::DT,
    //     constants::SAMPLES_COUNT,
    //     constants::WARM_UP_COUNT,
    //     constants::TRAIN_SET_LEN,
    //     constants::K_FOLDS_COUNT,
    //     constants::TRAIN_SUBSET_LEN,
    //     constants::MASKS_COUNT,
    //     params.eps,
    //     params.k,
    //     params.alpha,
    //     params.beta,
    //     params.tau_0,
    //     params.omega_0,
    //     params.gamma,
    //     params.A,
    //     params.eta,
    //     constants::XOR_K_STEPS
    // ));
    string_params.push_str(string);
    string_params.push('\n');

    file.write(string_params.as_bytes());
}