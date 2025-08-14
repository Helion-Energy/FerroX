#include "LCCircuit.H"

LCCircuit::LCCircuit()
{
    // Baseline defaults (you requested these)
    m_L = 22.0e-6;     // 22 uH
    m_C = 48.0e-6;     // 48 uF
    m_voltage = 10.0e3; // 10 kV initial capacitor voltage (V)
    m_current = 0.0;    // initial current (A)
    m_time = 0.0;
}

void LCCircuit::Initialize()
{
    amrex::ParmParse pp_circuit("circuit");

    int enable_lc_test = 0;
    pp_circuit.query("enable_lc_test", enable_lc_test);

    if (enable_lc_test) {
        amrex::Print() << "LC circuit enabled\n";
        pp_circuit.query("lc_inductance", m_L);
        pp_circuit.query("lc_capacitance", m_C);
        pp_circuit.query("lc_initial_voltage", m_voltage);
        pp_circuit.query("lc_initial_current", m_current);
        
	amrex::Print() << "LCCircuit parameters:\n";
        amrex::Print() << "Inductance (L): " << m_L << " H\n";
        amrex::Print() << "Capacitance (C): " << m_C << " F\n";
        amrex::Print() << "Initial Voltage: " << m_voltage << " V\n";
        amrex::Print() << "Initial Current: " << m_current << " A\n";

    }
}

void LCCircuit::UpdateTimeStep(amrex::Real dt, amrex::Real tcad_diode_voltage)
{
    if (dt <= 0.0) {
        // nothing to do
        return;
    }

    // Save initial state
    amrex::Real V0 = m_voltage;
    amrex::Real I0 = m_current;

    // Note: tcad_diode_voltage is treated as constant for all RK4 substeps (staggered coupling).
    auto deriv = [&](amrex::Real V, amrex::Real I) -> std::pair<amrex::Real, amrex::Real> {
        // dV/dt = I / C
        amrex::Real dVdt = I / m_C;
        // dI/dt = (-V - V_diode) / L  (loop equation: L dI/dt + V + V_diode = 0)
        amrex::Real dIdt = (-V - tcad_diode_voltage) / m_L;
        return std::make_pair(dVdt, dIdt);
    };

    // RK4 coefficients
    auto k1 = deriv(V0, I0);

    amrex::Real V_k2 = V0 + 0.5 * dt * k1.first;
    amrex::Real I_k2 = I0 + 0.5 * dt * k1.second;
    auto k2 = deriv(V_k2, I_k2);

    amrex::Real V_k3 = V0 + 0.5 * dt * k2.first;
    amrex::Real I_k3 = I0 + 0.5 * dt * k2.second;
    auto k3 = deriv(V_k3, I_k3);

    amrex::Real V_k4 = V0 + dt * k3.first;
    amrex::Real I_k4 = I0 + dt * k3.second;
    auto k4 = deriv(V_k4, I_k4);

    // Combine increments
    amrex::Real dV = (dt / 6.0) * (k1.first + 2.0 * k2.first + 2.0 * k3.first + k4.first);
    amrex::Real dI = (dt / 6.0) * (k1.second + 2.0 * k2.second + 2.0 * k3.second + k4.second);

    m_voltage = V0 + dV;
    m_current = I0 + dI;

    m_time += dt;
}

amrex::Real LCCircuit::GetVoltage() const
{
    return m_voltage;
}

void LCCircuit::SetCurrent(amrex::Real current)
{
    m_current = current;
}

amrex::Real LCCircuit::GetCurrent() const
{
    return m_current;
}

