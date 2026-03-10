use crate::constants;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Normal;

const eps: f64 = 0.05; // epsilon = (1 / w_LPF)
const k: f64 = 0.8;
const alpha: f64 = 0.4;
const beta: f64 = 0.5;
const tau_0: f64 = 1.;


pub struct Params {
    pub eps: f64,
    pub k: f64,
    pub alpha: f64,
    pub beta: f64,
    pub tau_0: f64,
    pub Y0: f64,
    pub T_cc: f64,
    pub Theta: f64,
    pub NODES: usize,
    pub MAX_HIST_LEN: usize,
    pub TOTAL_OUTPUT_NODES: usize,
    pub TOTAL_TIME_TO_COMPUTE: f64,
    pub TOTAL_ITERS_TO_COMPUTE: usize,
    pub CALLBACK_CONSOLE_OUTPUT_INTERVAL: usize,
    pub DT: f64,
    pub noise_rng: StdRng,
    pub normal_distr: Normal<f64>,
    pub D_STEPS_COUNT: usize,
    pub XC_D_STEPS_COUNT: usize,
    pub every_second: bool,
    pub TRAINING_NODES: usize
}

impl Params {
    pub fn defaults() -> Self {
        Self {
            eps: eps, // epsilon = (1 / w_LPF)
            k: k,
            alpha: alpha,
            beta: beta,
            tau_0: tau_0,
            Y0: alpha,
            T_cc: 1.2 * tau_0,
            Theta: 0.12,
            NODES: (10. * tau_0) as usize,
            MAX_HIST_LEN: (4. * (tau_0 / constants::DEFAULT_DT)) as usize,
            TOTAL_OUTPUT_NODES: (10. * tau_0) as usize * constants::SAMPLES_COUNT,
            TOTAL_TIME_TO_COMPUTE: 1.2 * tau_0 * constants::SAMPLES_COUNT as f64,
            TOTAL_ITERS_TO_COMPUTE: (1.2 * tau_0 * constants::SAMPLES_COUNT as f64 / constants::DEFAULT_DT) as usize,
            CALLBACK_CONSOLE_OUTPUT_INTERVAL: ((1.2 * tau_0 * constants::SAMPLES_COUNT as f64 / constants::DEFAULT_DT) * (constants::CALLBACK_CONSOLE_OUTPUT_INTERVAL_PERCENT / 100.)) as usize,
            DT: constants::DEFAULT_DT,
            noise_rng: StdRng::seed_from_u64(2025),
            normal_distr: Normal::new(0., 2.2e-4).unwrap(),
            D_STEPS_COUNT: 500,
            XC_D_STEPS_COUNT: 500,
            every_second: false,
            TRAINING_NODES: (10. * tau_0) as usize
        }
    }

    pub fn update(&mut self, args: &crate::Args) {
        if !args.every_second {
            if args.constant_theta {
                self.NODES = (10. * self.tau_0) as usize;
                self.TRAINING_NODES = self.NODES;
            } else {
                self.NODES = 100;
                self.TRAINING_NODES = self.NODES;
                self.tau_0 = self.NODES as f64 * self.Theta / 1.2;
            }
        } else {
            self.NODES = 200;
            self.TRAINING_NODES = 100;
            self.tau_0 = self.NODES as f64 * self.Theta / 1.2;
        }

        self.DT = self.Theta / 16.;
        // self.tau_0 = self.NODES as f64 * self.Theta / 1.2;
        self.T_cc = self.NODES as f64 * self.Theta;
        self.MAX_HIST_LEN = (4. * (self.tau_0 / self.DT)) as usize;
        self.TOTAL_OUTPUT_NODES = self.NODES * constants::SAMPLES_COUNT;
        self.TOTAL_TIME_TO_COMPUTE = self.NODES as f64 * self.Theta * constants::SAMPLES_COUNT as f64;
        self.TOTAL_ITERS_TO_COMPUTE = (self.TOTAL_TIME_TO_COMPUTE / self.DT) as usize;
        self.CALLBACK_CONSOLE_OUTPUT_INTERVAL = (self.TOTAL_ITERS_TO_COMPUTE as f64 * (constants::CALLBACK_CONSOLE_OUTPUT_INTERVAL_PERCENT / 100.)) as usize;
    }
}