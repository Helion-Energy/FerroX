#ifndef LCCIRCUIT_H
#define LCCIRCUIT_H

#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>

class LCCircuit {
public:
    // Default constructor
    LCCircuit();

    // Initialize the circuit parameters from the input file (section "circuit")
    void Initialize();

    // Update the circuit state using RK4 for the series L-C with a diode drop (frozen during substeps).
    // dt: timestep (s)
    // tcad_diode_voltage: diode voltage from TCAD (V) — treated as constant across the RK4 sub-steps
    void UpdateTimeStep(amrex::Real dt, amrex::Real tcad_diode_voltage);

    // Get current capacitor voltage (V)
    amrex::Real GetVoltage() const;

    // Set the circuit current (A) — user may want to seed the solver
    void SetCurrent(amrex::Real current);

    // Get the circuit current (A)
    amrex::Real GetCurrent() const;

    // Optional: get simulation time tracked by the circuit
    amrex::Real GetTime() const { return m_time; }

private:
    // Circuit parameters
    amrex::Real m_L;      // Henries
    amrex::Real m_C;      // Farads

    // State variables
    amrex::Real m_voltage; // Voltage across capacitor (V)
    amrex::Real m_current; // Series current (A)
    amrex::Real m_time;    // simulation time (s)
};

#endif // LCCIRCUIT_H

