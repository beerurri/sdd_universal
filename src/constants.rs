// pub const DT: f64 = 0.001;
pub const DEFAULT_DT: f64 = 0.001;

pub const SAMPLES_COUNT: usize = 5000;

pub const WARM_UP_COUNT: usize = 1500;

pub const MC_TEST_SET_LEN: usize = 200;
pub const SANTA_TEST_SET_LEN: usize = 1000;
pub const NARMA_TEST_SET_LEN: usize = 200;

pub const MC_TRAIN_SET_LEN: usize = SAMPLES_COUNT - WARM_UP_COUNT - MC_TEST_SET_LEN;
pub const SANTA_TRAIN_SET_LEN: usize = SAMPLES_COUNT - WARM_UP_COUNT - SANTA_TEST_SET_LEN;
pub const NARMA_TRAIN_SET_LEN: usize = SAMPLES_COUNT - WARM_UP_COUNT - NARMA_TEST_SET_LEN;

pub const MC_K_FOLDS_COUNT: usize = 50;
pub const SANTA_GAP_SET_LEN: usize = 500;
pub const SANTA_K_STEPS: usize = 1;

pub const FEATURES_OUT: usize = 1;

// pub const K_FOLDS_COUNT: usize = 50;
// pub const TRAIN_SUBSET_LEN: usize = TRAIN_SET_LEN / K_FOLDS_COUNT;
pub const MASKS_COUNT: usize = 50;

pub const CALLBACK_CONSOLE_OUTPUT_INTERVAL_PERCENT: f64 = 20.;

pub const MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL_PERCENT: f64 = 10.;
pub const MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL: usize = (MC_K_FOLDS_COUNT as f64 * MC_XVALIDATION_CONSOLE_OUTPUT_INTERVAL_PERCENT / 100.) as usize;