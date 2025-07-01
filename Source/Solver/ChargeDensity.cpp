#include "ChargeDensity.H"
#include "DerivativeAlgorithm.H"

// Approximation to the Fermi-Dirac Integral of Order 1/2
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real FD_half(const amrex::Real eta)
{
    amrex::Real nu = std::pow(eta, 4.0) + 50.0 + 33.6 * eta * (1.0 - 0.68 * exp(-0.17 * std::pow((eta + 1.0), 2.0)));
    amrex::Real xi = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu, 3./8.));
    amrex::Real integral = std::pow(exp(-eta) + xi, -1.0);
    return integral;
}

// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
		const MultiFab& MaterialMask)
{
    amrex::Print() << "Calculating steady-state carrier distribution." << "\n";

    //Define acceptor and donor multifabs for doping and fill them with zero.
    MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    acceptor_den.setVal(0.);
    donor_den.setVal(0.);

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             if (mask(i,j,k) >= 2.0) {
      
                //Following: http://dx.doi.org/10.1063/1.4825209

                amrex::Real Ef = 0.0;
                amrex::Real Eg = bandgap;
                amrex::Real Chi = affinity;
                amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q;
                amrex::Real Ec_corr = -q*(phi(i,j,k) - phi_ref) - Chi*q;
                amrex::Real Ev_corr = Ec_corr - q*Eg; 

                //g_A is the acceptor ground state degeneracy factor and is equal to 4 
                //because in most semiconductors each acceptor level can accept one hole of either spin 
                //and the impurity level is doubly degenerate as a result of the two degenerate valence bands 
                //(heavy hole and light hole bands) at the \Gamma point.

                //g_D is the donor ground state degeneracy factor and is equal to 2
                //because a donor level can accept one electron with either spin or can have no electron when filled.

                amrex::Real g_A = 4.0;
                amrex::Real g_D = 2.0;

                amrex::Real Ea = acceptor_ionization_energy;  
                amrex::Real Ed = donor_ionization_energy; 

                amrex::Real Na, Nd;

                if (mask(i,j,k) == 2.0) {//intrinsic
                   Na = 0.0;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 3.0) { // p-type
                   Na = acceptor_doping;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 4.0) { // n-type
                   Na = 0.0;
                   Nd = donor_doping;
                }
                  
                if(use_Fermi_Dirac == 1){
                  //Fermi-Dirac

                  Real eta_n = -(Ec_corr - q*Ef)/(kb*T);
                  Real eta_p = -(q*Ef - Ev_corr)/(kb*T);
                  e_den_arr(i,j,k) = Nc*FD_half(eta_n);
                  hole_den_arr(i,j,k) = Nv*FD_half(eta_p);
         
                  acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                  } else {

                  //Maxwell-Boltzmann
                  e_den_arr(i,j,k) =    Nc*exp( -(Ec_corr - q*Ef) / (kb*T) );
                  hole_den_arr(i,j,k) = Nv*exp( -(q*Ef - Ev_corr) / (kb*T) );
               
                  acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                }

		charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }


// --- Bernoulli Function Implementation ---
// It's defined as Bern(x) = x / (exp(x) - 1).
// Special care is needed for x close to 0 to avoid division by zero (use Taylor expansion).
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real Bern(amrex::Real x)
{
    // Use a small epsilon for robustness around x=0
    if (amrex::Math::abs(x) < 1.0e-4) {
        // Taylor expansion for small x: 1 - x/2 + x^2/12 - x^4/720 + ...
        return 1.0 - x/2.0; // + x*x/12.0;
    } 
    // Handle large positive arguments
    else if (x > 50.0) {
        // For large x: x/(exp(x)-1) ≈ x/exp(x) ≈ 0
        return 0.0;
    }
    // Handle large negative arguments  
    else if (x < -50.0) {
        // For large negative x: x/(exp(x)-1) ≈ x/(-1) = -x
        return -x;
    }
    else {
        return x / (exp(x) - 1.0);
    }
}
// --- CalculateDriftDiffusionCurrents ---
void CalculateDriftDiffusionCurrents(
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jn, // OUT: Electron current density components (x,y,z)
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jp, // OUT: Hole current density components (x,y,z)
    const amrex::MultiFab& e_den,                      // IN: Electron density
    const amrex::MultiFab& p_den,                      // IN: Hole density
    const amrex::MultiFab& MaterialMask,               // IN: Material mask
    const amrex::MultiFab& PoissonPhi,                 // IN: Electric potential
    const amrex::Geometry& geom                        // IN: Simulation geometry
)
{
    const amrex::Real kBT_over_q = kb * T / q;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Create effective potentials with proper ghost cells
    MultiFab e_potential(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 1);
    MultiFab p_potential(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 1);
    e_potential.setVal(0.);
    p_potential.setVal(0.);

    // Compute quasi-Fermi potentials
    Compute_Effective_Potentials(PoissonPhi, e_den, p_den, e_potential, p_potential, MaterialMask, geom);

    // Get domain boundaries
    const Box& domain = geom.Domain();

    for (amrex::MFIter mfi(e_den, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const& e_den_arr = e_den.const_array(mfi);
        amrex::Array4<amrex::Real const> const& p_den_arr = p_den.const_array(mfi);
        amrex::Array4<amrex::Real const> const& phi_arr = PoissonPhi.const_array(mfi);
        amrex::Array4<amrex::Real const> const& phi_n_arr = e_potential.const_array(mfi);
        amrex::Array4<amrex::Real const> const& phi_p_arr = p_potential.const_array(mfi);
        amrex::Array4<Real const> const& mask = MaterialMask.const_array(mfi);

        amrex::Array4<amrex::Real> const& Jnx_arr = Jn[0].array(mfi);
        amrex::Array4<amrex::Real> const& Jny_arr = Jn[1].array(mfi);
        amrex::Array4<amrex::Real> const& Jnz_arr = Jn[2].array(mfi);
        amrex::Array4<amrex::Real> const& Jpx_arr = Jp[0].array(mfi);
        amrex::Array4<amrex::Real> const& Jpy_arr = Jp[1].array(mfi);
        amrex::Array4<amrex::Real> const& Jpz_arr = Jp[2].array(mfi);

        // Material parameters
        amrex::Real mu_n = electron_mobility;
        amrex::Real mu_p = hole_mobility;
        amrex::Real D_n = electron_diffusion_coefficient;
        amrex::Real D_p = hole_diffusion_coefficient;

	amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
        {
            // Initialize all current components to zero
            Jnx_arr(i, j, k) = 0.0;
            Jny_arr(i, j, k) = 0.0;
            Jnz_arr(i, j, k) = 0.0;
            Jpx_arr(i, j, k) = 0.0;
            Jpy_arr(i, j, k) = 0.0;
            Jpz_arr(i, j, k) = 0.0;
        
            // Only compute currents in semiconductor regions
            if (mask(i,j,k) >= 2.0) {
        
                // --- Calculate J_x (current across faces normal to x-axis) ---
                if (i <= domain.bigEnd(0)) { 
                    if (mask(i+1,j,k) >= 2.0) {
                        amrex::Real dPhi_n = phi_n_arr(i+1, j, k) - phi_n_arr(i, j, k);
                        amrex::Real arg_n = dPhi_n / kBT_over_q;
                
                        amrex::Real dPhi_p = phi_p_arr(i+1, j, k) - phi_p_arr(i, j, k);
                        amrex::Real arg_p = dPhi_p / kBT_over_q;
                
                        // Electron current
                        Jnx_arr(i, j, k) = q * D_n / dx[0] * (e_den_arr(i+1,j,k) * Bern(arg_n) - e_den_arr(i,j,k) * Bern(-arg_n));
                        
                        // Hole current
                        Jpx_arr(i, j, k) = -q * D_p / dx[0] * (p_den_arr(i+1,j,k) * Bern(-arg_p) - p_den_arr(i,j,k) * Bern(arg_p));
                    }
                }
                
                // --- Calculate J_y (current across faces normal to y-axis) ---
                if (j <= domain.bigEnd(1)) {
                    if (mask(i,j+1,k) >= 2.0) {
                        amrex::Real dPhi_n_y = phi_n_arr(i, j+1, k) - phi_n_arr(i, j, k);
                        amrex::Real arg_n_y = dPhi_n_y / kBT_over_q;
                
                        amrex::Real dPhi_p_y = phi_p_arr(i, j+1, k) - phi_p_arr(i, j, k);
                        amrex::Real arg_p_y = dPhi_p_y / kBT_over_q;
                
                        // Electron current
                        Jny_arr(i, j, k) = q * D_n / dx[1] * (e_den_arr(i,j+1,k) * Bern(arg_n_y) - e_den_arr(i,j,k) * Bern(-arg_n_y));
                        
                        // Hole current
                        Jpy_arr(i, j, k) = -q * D_p / dx[1] * (p_den_arr(i,j+1,k) * Bern(-arg_p_y) - p_den_arr(i,j,k) * Bern(arg_p_y));
                    }
                }
                
                // --- Calculate J_z (current across faces normal to z-axis) ---
                if (k <= domain.bigEnd(2)) {
                    if (mask(i,j,k+1) >= 2.0) {
                        amrex::Real dPhi_n_z = phi_n_arr(i, j, k+1) - phi_n_arr(i, j, k);
                        amrex::Real arg_n_z = dPhi_n_z / kBT_over_q;
                
                        amrex::Real dPhi_p_z = phi_p_arr(i, j, k+1) - phi_p_arr(i, j, k);
                        amrex::Real arg_p_z = dPhi_p_z / kBT_over_q;
                
                        // Electron current
                        Jnz_arr(i, j, k) = q * D_n / dx[2] * (e_den_arr(i,j,k+1) * Bern(arg_n_z) - e_den_arr(i,j,k) * Bern(-arg_n_z));
                        
                        // Hole current
                        Jpz_arr(i, j, k) = -q * D_p / dx[2] * (p_den_arr(i,j,k+1) * Bern(-arg_p_z) - p_den_arr(i,j,k) * Bern(arg_p_z));
                        //if(i == 1 && j == 1 && (k == domain.bigEnd(2) || k == domain.bigEnd(2) - 1)){
                        //    amrex::Real p_den_k = p_den_arr(i,j,k);
                        //    amrex::Real p_den_kp1 = p_den_arr(i,j,k+1); // THIS IS THE CRITICAL VALUE TO CHECK
                        //
                        //    amrex::Print() << "k = " << k << ", p_den_arr(k) = " << p_den_k << "\n";
                        //    amrex::Print() << "k = " << k << ", p_den_arr(k+1) = " << p_den_kp1 << "\n";
                        //    amrex::Print() << "k = " << k << ", phi_arr(k) = " << phi_p_arr(i,j,k) << "\n";
                        //    amrex::Print() << "k = " << k << ", phi_arr(k+1) = " << phi_p_arr(i,j,k+1) << "\n";
                        //    amrex::Print() << "k = " << k << ", Bern(-arg_p_z)  = " << Bern(-arg_p_z) << "\n";
                        //    amrex::Print() << "k = " << k << ", Bern(arg_p_z)  = " << Bern(arg_p_z) << "\n";
                        //}
		    }
                } 
            }
        });
    }

    // Fill boundary conditions for current arrays
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Jn[d].FillBoundary(geom.periodicity());
        Jp[d].FillBoundary(geom.periodicity());
    }
}

// Compute rho in SC region for given phi
void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                Array<MultiFab, AMREX_SPACEDIM> &Jn,
                Array<MultiFab, AMREX_SPACEDIM> &Jp,
                MultiFab&      e_den,
                MultiFab&      p_den,
                MultiFab&      acceptor_den,
                MultiFab&      donor_den,
                MultiFab& MaterialMask,
                const Geometry& geom)
{
//    amrex::Print() << "Calculating carrier transport using Drift-Diffusion model." << "\n";

    // First, calculate the current components and store them in Jn and Jp
    CalculateDriftDiffusionCurrents(Jn, Jp, e_den, p_den, MaterialMask, PoissonPhi, geom);

    // Get cell spacing from geometry
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Get domain boundaries
    const Box& domain = geom.Domain();
    const int domain_lo_z = domain.smallEnd(2);
    const int domain_hi_z = domain.bigEnd(2);

    // Get the intrinsic carrier concentration
    amrex::Real ni_val = intrinsic_carrier_concentration;
    amrex::Real ni_sq_val = ni_val * ni_val;

    // SRH recombination parameters
    amrex::Real tau_n_val = electron_lifetime; // Electron lifetime
    amrex::Real tau_p_val = hole_lifetime; // Hole lifetime

    // Loop over grids (boxes) in the MultiFab for updating densities
    for (amrex::MFIter mfi(e_den, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();

        // Get Array4 views for densities
        amrex::Array4<amrex::Real> const& e_den_arr = e_den.array(mfi);
        amrex::Array4<amrex::Real> const& p_den_arr = p_den.array(mfi);
        amrex::Array4<amrex::Real> const& charge_den_arr = rho.array(mfi);
        amrex::Array4<amrex::Real> const& phi = PoissonPhi.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);

        // Get Array4 views for current components
        amrex::Array4<amrex::Real const> const& Jnx_arr = Jn[0].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jny_arr = Jn[1].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jnz_arr = Jn[2].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpx_arr = Jp[0].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpy_arr = Jp[1].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpz_arr = Jp[2].const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
        {
            if (mask(i,j,k) >= 2.0) {

                // Check if we're at domain boundaries
                bool at_left_contact = (k == domain_lo_z);
                bool at_right_contact = (k == domain_hi_z);

                // ONLY UPDATE INTERIOR POINTS - NOT CONTACTS
                if (!at_left_contact && !at_right_contact) {
                    // Interior points - update using drift-diffusion equations

                    // --- Calculate Current Divergence Safely ---
                    amrex::Real div_Jn = 0.0;
                    amrex::Real div_Jp = 0.0;

		    // X-direction divergence
                    if (i == domain.smallEnd(0)) {
                        div_Jn += Jnx_arr(i, j, k) / dx[0];
                        div_Jp += Jpx_arr(i, j, k) / dx[0];
                    } else if (i == domain.bigEnd(0)) {
                        div_Jn += -Jnx_arr(i-1, j, k) / dx[0];
                        div_Jp += -Jpx_arr(i-1, j, k) / dx[0];
                    } else {
                        div_Jn += (Jnx_arr(i, j, k) - Jnx_arr(i-1, j, k)) / dx[0];
                        div_Jp += (Jpx_arr(i, j, k) - Jpx_arr(i-1, j, k)) / dx[0];
                    }
                    
                    // Y-direction divergence
                    if (j == domain.smallEnd(1)) {
                        div_Jn += Jny_arr(i, j, k) / dx[1];
                        div_Jp += Jpy_arr(i, j, k) / dx[1];
                    } else if (j == domain.bigEnd(1)) {
                        div_Jn += -Jny_arr(i, j-1, k) / dx[1];
                        div_Jp += -Jpy_arr(i, j-1, k) / dx[1];
                    } else {
                        div_Jn += (Jny_arr(i, j, k) - Jny_arr(i, j-1, k)) / dx[1];
                        div_Jp += (Jpy_arr(i, j, k) - Jpy_arr(i, j-1, k)) / dx[1];
                    }
                    
                    // Z-direction divergence
                    if (k == domain.smallEnd(2)) {
                        div_Jn += Jnz_arr(i, j, k) / dx[2];
                        div_Jp += Jpz_arr(i, j, k) / dx[2];
                    } else if (k == domain.bigEnd(2)) {
                        div_Jn += -Jnz_arr(i, j, k-1) / dx[2];
                        div_Jp += -Jpz_arr(i, j, k-1) / dx[2];
                    } else {
                        div_Jn += (Jnz_arr(i, j, k) - Jnz_arr(i, j, k-1)) / dx[2];
                        div_Jp += (Jpz_arr(i, j, k) - Jpz_arr(i, j, k-1)) / dx[2];
                    }
                    
		    // --- Calculate SRH Net Recombination Rate ---
                    amrex::Real current_n = e_den_arr(i, j, k);
                    amrex::Real current_p = p_den_arr(i, j, k);

                    // Ensure positive densities for recombination calculation
                    current_n = amrex::max(current_n, 1.0e10);
                    current_p = amrex::max(current_p, 1.0e10);

                    amrex::Real SRH_numerator = (current_n * current_p) - ni_sq_val;
                    amrex::Real SRH_denominator = tau_p_val * (current_n + ni_val) + tau_n_val * (current_p + ni_val);

                    amrex::Real R_SRH = 0.0;
                    if (SRH_denominator > 1.0e-30) {
                        R_SRH = SRH_numerator / SRH_denominator;
                    }

		    amrex::Real recomb_term = (use_srh_recombination == 1) ? R_SRH : 0.0;

                    // --- Update Carrier Densities ---
                    // Continuity equations:
                    // ∂n/∂t = (1/q) * ∇·Jn - R
                    // ∂p/∂t = -(1/q) * ∇·Jp - R
                    e_den_arr(i, j, k) += dt * ((1.0/q) * div_Jn - recomb_term);
                    p_den_arr(i, j, k) += dt * ((-1.0/q) * div_Jp - recomb_term);

                    // Ensure carrier densities remain positive
                    e_den_arr(i, j, k) = amrex::max(e_den_arr(i, j, k), 1.0e10);
                    p_den_arr(i, j, k) = amrex::max(p_den_arr(i, j, k), 1.0e10);
                }

                // --- Assume Complete Ionization For Now ---
		// Partial Ionization Model would modify acceptor_den_arr(i,j,k) and donor_den_arr(i,j,k) based on local phi.

		// --- Update Total Charge Density ---
                charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));
            }
        });
    }
/*
    // **NOW SET CONTACT BOUNDARY CONDITIONS AFTER THE MAIN LOOP**
    for (amrex::MFIter mfi(e_den); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(1);
        amrex::Array4<amrex::Real> const& e_den_arr = e_den.array(mfi);
        amrex::Array4<amrex::Real> const& p_den_arr = p_den.array(mfi);
        amrex::Array4<amrex::Real> const& charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            if (mask(i,j,k) >= 2.0) {
                //bool at_left_contact = (k == domain_lo_z);
                //bool at_right_contact = (k == domain_hi_z);

                if (k <= domain_lo_z) {
                    // Left contact - N-type boundary condition
                    e_den_arr(i, j, k) = 0.5*donor_doping + std::sqrt(std::pow(0.5*donor_doping, 2.0) + ni_sq_val);
                    p_den_arr(i, j, k) = ni_sq_val / e_den_arr(i, j, k);
                    
                    // Update charge density for contact
                    charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) + donor_den_arr(i,j,k));
                }
                else if (k >= domain_hi_z) {
                    // Right contact - P-type boundary condition
                    p_den_arr(i, j, k) = 0.5*acceptor_doping + std::sqrt(std::pow(0.5*acceptor_doping, 2.0) + ni_sq_val);
                    e_den_arr(i, j, k) = ni_sq_val / p_den_arr(i, j, k);
                    
                    // Update charge density for contact
                    charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k));
                }
            }
        });
    }
*/
    // **NOW SET CONTACT BOUNDARY CONDITIONS AFTER THE MAIN LOOP**
    for (amrex::MFIter mfi(e_den); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.growntilebox(1);
        amrex::Array4<amrex::Real> e_den_arr = e_den.array(mfi);
        amrex::Array4<amrex::Real> p_den_arr = p_den.array(mfi);
        amrex::Array4<amrex::Real> charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);
    
        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            // Check if we are at a contact boundary in z
            if (k <= domain_lo_z || k >= domain_hi_z) {
                // Check if the material is a semiconductor
                if (mask(i,j,k) >= 2.0) {
                    // Now check the material type to determine the BC and use local doping
                    if (mask(i,j,k) == 4.0 || mask(i,j,k) == 6.0) { // n-type or n++
                        // Apply N-type BC using local donor doping
                        amrex::Real local_doping = donor_den_arr(i, j, k);
                        e_den_arr(i, j, k) = 0.5*local_doping + std::sqrt(std::pow(0.5*local_doping, 2.0) + ni_sq_val);
                        p_den_arr(i, j, k) = ni_sq_val / e_den_arr(i, j, k);
    
                        // Update charge density for contact
                        charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) + local_doping);
                    } else if (mask(i,j,k) == 3.0 || mask(i,j,k) == 5.0) { // p-type or p++
                        // Apply P-type BC using local acceptor doping
                        amrex::Real local_doping = acceptor_den_arr(i, j, k);
                        p_den_arr(i, j, k) = 0.5*local_doping + std::sqrt(std::pow(0.5*local_doping, 2.0) + ni_sq_val);
                        e_den_arr(i, j, k) = ni_sq_val / p_den_arr(i, j, k);
    
                        // Update charge density for contact
                        charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) - local_doping);
                    }
                }
            }
        });
    }

    // Fill ghost cells for updated multifabs
    e_den.FillBoundary(geom.periodicity());
    p_den.FillBoundary(geom.periodicity());
    rho.FillBoundary(geom.periodicity());
}

// Approximation to the inverse of the Fermi-Dirac Integral of Order 1/2
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real Inverse_FD_half(amrex::Real u)
{
    // Handle edge cases
    if (u <= 0.0) return -1000.0; // Very negative eta (empty states)
    if (u < 1.0e-10) return std::log(u); // Maxwell-Boltzmann limit for small u
    
    amrex::Real sqrt_pi = std::sqrt(3.14159265359);
    amrex::Real nu = std::pow((3.0 * sqrt_pi * u / 4.0), 2.0 / 3.0);

    amrex::Real log_term;
    if (std::abs(u*u - 1.0) < 1.0e-10) {
        // Near u = 1, use series expansion or alternative formula
        // For u close to 1: log_term ≈ -0.5 * (u - 1) (first-order approximation)
        log_term = -0.5 * (u - 1.0);
    } else {
        log_term = -std::log(u) / (u*u - 1.0);
    }
    
    amrex::Real denom = 1.0 + std::pow(0.24 + 1.08 * nu, -2.0);
    amrex::Real eta = log_term + nu / denom;

    return eta;
}

void Compute_Effective_Potentials(const MultiFab& PoissonPhi,
                                  const MultiFab& e_den,
                                  const MultiFab& p_den,
                                  MultiFab& e_potential,
                                  MultiFab& p_potential,
                                  const MultiFab& MaterialMask,
                                  const Geometry& geom)
{
    // Material parameters
    amrex::Real Nc_val = Nc;
    amrex::Real Nv_val = Nv;
    amrex::Real Eg = bandgap;
    amrex::Real Chi = affinity;
    amrex::Real Efn = 0.0;
    amrex::Real Efp = 0.0;

    // Constants
    amrex::Real kT = kb * T;
    amrex::Real kT_q = kT / q;

    // Reference potential: intrinsic Fermi level relative to vacuum
    amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kT_q*log(Nc_val/Nv_val);

    // Intrinsic carrier concentration
    Real ni_val = intrinsic_carrier_concentration;

    // Get domain boundaries
    const Box& domain = geom.Domain();
    const int domain_lo_z = domain.smallEnd(2);
    const int domain_hi_z = domain.bigEnd(2);

    // Loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        //const Box& bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real const>& phi = PoissonPhi.array(mfi);
        const Array4<Real const>& n_arr = e_den.const_array(mfi);
        const Array4<Real const>& p_arr = p_den.const_array(mfi);
        const Array4<Real>& phi_n_eff = e_potential.array(mfi);
        const Array4<Real>& phi_p_eff = p_potential.array(mfi);
        const Array4<Real const>& mask = MaterialMask.const_array(mfi);

        // Main calculation using Fermi-Dirac effective fields
        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
        {

	      // Get local values
              Real phi_val = phi(i,j,k);

              // Calculate band edges in Joules
              Real Ec_J = -q * (phi_val - phi_ref) - q * Chi;
              Real Ev_J = Ec_J - q * Eg;

              Real eta_n = (Efn - Ec_J)/kT; //Inverse_FD_half(u_n);  // (Efn - Ec)/kT
              Real eta_p = (Ev_J - Efp)/kT; //Inverse_FD_half(u_p);  // (Ev - Efp)/kT


              // Calculate degeneracy factors
              // γ = F_{1/2}(η) / exp(η)
              Real gamma_n, gamma_p;

              if (eta_n < -5.0) {
                  gamma_n = 1.0; // Maxwell-Boltzmann limit
              } else {
                  Real F_half_n = FD_half(eta_n);
                  gamma_n = F_half_n / std::exp(eta_n);
              }

              if (eta_p < -5.0) {
                  gamma_p = 1.0; // Maxwell-Boltzmann limit
              } else {
                  Real F_half_p = FD_half(eta_p);
                  gamma_p = F_half_p / std::exp(eta_p);
              }

              // Calculate intrinsic Fermi energy (Equation 11 from Charon manual)
              Real Ei_J = q*phi_ref - Chi*q - q*phi_val - Eg*q/2.0 - (kT*q/2.0) * log((Nc_val*gamma_n)/(Nv_val*gamma_p));
    
              // Calculate effective potential terms (Equation 10)
              // The effective fields are: F_n,eff = -∇(φ_n,eff) and F_p,eff = -∇(φ_p,eff)
              // where:
              // φ_n,eff corresponds to: -(Ei - ΔEg/2 - (kT/2)*ln(γn*γp))/q
              // φ_p,eff corresponds to: -(Ei + ΔEg/2 + (kT/2)*ln(γn*γp))/q

              Real degeneracy_term = (kT/2.0) * log(gamma_n * gamma_p);

	      //Bandgap narrowing DeltaEg is in eV. Multiplication by q is to convert it into J to match other terms
              Real phi_n_eff_val = -(Ei_J - q*DeltaEg/2.0 - degeneracy_term)/q;
              Real phi_p_eff_val = -(Ei_J + q*DeltaEg/2.0 + degeneracy_term)/q;

              // Store the effective potentials
              // The Scharfetter-Gummel method will use gradients of these potentials
              phi_n_eff(i,j,k) = phi_n_eff_val;
              phi_p_eff(i,j,k) = phi_p_eff_val;
        });
    }

    e_potential.FillBoundary(geom.periodicity());
    p_potential.FillBoundary(geom.periodicity());
}
