use crate::params;

/// Right-hand side function
/// 
/// `t`: current problem's time
/// 
/// `y`: current problem's `y` argument
/// 
/// `H`: 'get history' function's type alias
/// 
/// `t`, `y`, `H`, `history`
pub type F<'a> = &'a dyn Fn(f64, f64, &Vec<f64>, &mut params::Params, &Vec<f64>, &Vec<f64>) -> f64;

/// 'Get history' fucntion that returns historical `y` value at the moment (`t_curr` - `time_back`)
/// 
/// `t_curr`, `y_curr`, `history`
pub type H<'a> = &'a dyn Fn(f64, &Vec<f64>, &params::Params) -> f64;

/// Callback function (optional)
/// 
/// `counter`
pub type C<'a> = &'a dyn Fn(usize, &Vec<f64>, &params::Params) -> ();