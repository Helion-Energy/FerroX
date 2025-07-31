// This is the standalone function for calculating the impact ionization generation rate.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real calculate_impact_ionization_generation_rate(
    amrex::Real alpha_n,
    amrex::Real j_n,
    amrex::Real alpha_p,
    amrex::Real j_p,
    amrex::Real q
)
{
    // The formula is: G_ii = (alpha_n * J_n / q) + (alpha_p * J_p / q)
    // The units are: (1/m) * (A/m^2) / (C) = (1/m) * (C/s/m^2) / (C) = m^-3 s^-1
    // This value represents the number of electron-hole pairs generated per unit volume per second.
    return (alpha_n * j_n / q) + (alpha_p * j_p / q);
}

// --- Implementation of the Okuto-Crowell model for ionization coefficients ---
// This function calculates the ionization coefficients for electrons (alpha_n)
// and holes (alpha_p) based on the electric field magnitude and temperature.
// The coefficients are from Sentaurus Manual Table 75 for Silicon.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void calculate_ionization_coefficients(
    amrex::Real E_field_magnitude,
    amrex::Real temperature,
    amrex::Real& alpha_n,
    amrex::Real& alpha_p
)
{
    // Okuto-Crowell coefficients from Table 75 for Silicon.
    // The values have been converted from CGS units (cm, V) to SI units (m, V)
    const amrex::Real T0 = 300.0; // Reference temperature in K

    // Electron coefficients
    const amrex::Real a_n = 0.426;
    const amrex::Real b_n = 4.81e7;  // 4.81e5 V/cm -> 4.81e7 V/m
    const amrex::Real c_n = 3.05e-4; // K^-1
    const amrex::Real d_n = 6.86e-4; // K^-1
    const amrex::Real gamma_n = 1.0;
    const amrex::Real delta_n = 2.0;

    // Hole coefficients
    const amrex::Real a_p = 0.243;
    const amrex::Real b_p = 6.53e7;  // 6.53e5 V/cm -> 6.53e7 V/m
    const amrex::Real c_p = 5.35e-4; // K^-1
    const amrex::Real d_p = 5.67e-4; // K^-1
    const amrex::Real gamma_p = 1.0;
    const amrex::Real delta_p = 2.0;

    // The Okuto-Crowell model is not valid for a zero or very low electric field.
    // We set a minimum threshold to avoid division by zero or numerical instability.
    const amrex::Real E_min_threshold = 1.0e-10;

    if (E_field_magnitude > E_min_threshold) {
        // Okuto-Crowell formula (Eq. 406):
        // Electron ionization coefficient
        amrex::Real temp_term_n = 1.0 + c_n * (temperature - T0);
        amrex::Real exp_term_n = std::pow((b_n * (1.0 + d_n * (temperature - T0))) / E_field_magnitude, delta_n);
        alpha_n = a_n * temp_term_n * std::pow(E_field_magnitude, gamma_n) * std::exp(-exp_term_n);
        
        // Hole ionization coefficient
        amrex::Real temp_term_p = 1.0 + c_p * (temperature - T0);
        amrex::Real exp_term_p = std::pow((b_p * (1.0 + d_p * (temperature - T0))) / E_field_magnitude, delta_p);
        alpha_p = a_p * temp_term_p * std::pow(E_field_magnitude, gamma_p) * std::exp(-exp_term_p);

    } else {
        alpha_n = 0.0;
        alpha_p = 0.0;
    }
}

