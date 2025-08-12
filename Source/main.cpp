#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif

#ifdef AMREX_USE_SUNDIALS
#include <AMReX_TimeIntegrator.H>
#endif

#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "FerroX.H"
#include "Solver/ElectrostaticSolver.H"
#include "Solver/Initialization.H"
#include "Solver/ChargeDensity.H"
#include "Solver/TotalEnergyDensity.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Utils/SelectWarpXUtils/WarpXProfilerWrapper.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "Utils/FerroXUtils/FerroXUtil.H"




using namespace amrex;

using namespace FerroX;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    {
	    c_FerroX pFerroX;
            pFerroX.InitData();
            main_main(pFerroX);
    }
    amrex::Finalize();
    return 0;
}

void main_main (c_FerroX& rFerroX)
{

    BL_PROFILE("main()");

    Real total_step_strt_time = ParallelDescriptor::second();

    auto& rGprop = rFerroX.get_GeometryProperties();
    auto& geom = rGprop.geom;
    auto& ba = rGprop.ba;
    auto& dm = rGprop.dm;
    auto& is_periodic = rGprop.is_periodic;
    auto& prob_lo = rGprop.prob_lo;
    auto& prob_hi = rGprop.prob_hi;
    auto& n_cell = rGprop.n_cell;

    // read in inputs file
    InitializeFerroXNamespace(prob_lo, prob_hi);

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    MultiFab Gamma(ba, dm, Ncomp, Nghost);

    Array<MultiFab, AMREX_SPACEDIM> P_old;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_old[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> P_new;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_new[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> P_new_pre;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_new_pre[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_Landau;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_Landau[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_grad;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_grad[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_elec;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_elec[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_pre;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_pre[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_avg;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_avg[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> E;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        E[dir].define(ba, dm, Ncomp, 0);
    }

    //Drift-Diffusion
    Array<MultiFab, AMREX_SPACEDIM> Jn;
    Array<MultiFab, AMREX_SPACEDIM> Jp;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        Jn[dir].define(ba, dm, Ncomp, 1);
        Jp[dir].define(ba, dm, Ncomp, 1);
    }

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1);
    MultiFab PoissonPhi_Old(ba, dm, 1, 1);
    PoissonPhi_Old.setVal(0.);
    MultiFab PoissonPhi_Prev(ba, dm, 1, 1);
    MultiFab PhiErr(ba, dm, 1, 1);
    MultiFab Phidiff(ba, dm, 1, 1);
    MultiFab Ex(ba, dm, 1, 0);
    MultiFab Ey(ba, dm, 1, 0);
    MultiFab Ez(ba, dm, 1, 0);

    MultiFab carrier_den(ba, dm, 2, 1);  // nComp = 2
    MultiFab carrier_rhs(ba, dm, 2, 1);

    MultiFab hole_den(ba, dm, 1, 1);
    MultiFab e_den(ba, dm, 1, 1);
    MultiFab acceptor_den(ba, dm, 1, 1);
    MultiFab donor_den(ba, dm, 1, 1);
    MultiFab AcceptorDoping(ba, dm, 1, 1); //for initial function based doping profile
    MultiFab DonorDoping(ba, dm, 1, 1); //for initial function based doping profile
    MultiFab charge_den(ba, dm, 1, 1);
    MultiFab e_rhs(ba, dm, 1, 1);
    MultiFab p_rhs(ba, dm, 1, 1);
    MultiFab MaterialMask(ba, dm, 1, 1);
    MultiFab tphaseMask(ba, dm, 1, 1);
    MultiFab angle_alpha(ba, dm, 1, 0);
    MultiFab angle_beta(ba, dm, 1, 0);
    MultiFab angle_theta(ba, dm, 1, 0);

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_old[dir].setVal(0.);
        P_new[dir].setVal(0.);
        P_new_pre[dir].setVal(0.);
        GL_rhs[dir].setVal(0.);
        GL_rhs_pre[dir].setVal(0.);
        GL_rhs_avg[dir].setVal(0.);
        E[dir].setVal(0.);
        Jn[dir].setVal(0.);
        Jp[dir].setVal(0.);
    }

    carrier_den.setVal(0.);
    carrier_rhs.setVal(0.);
    e_den.setVal(0.);
    hole_den.setVal(0.);
    acceptor_den.setVal(0.);
    donor_den.setVal(0.);
    AcceptorDoping.setVal(0.);
    DonorDoping.setVal(0.);
    e_rhs.setVal(0.);
    p_rhs.setVal(0.);
    PoissonPhi.setVal(0.);
    PoissonRHS.setVal(0.);
    tphaseMask.setVal(0.);
    angle_alpha.setVal(0.);
    angle_beta.setVal(0.);
    angle_theta.setVal(0.);

    //Initialize material mask
    //InitializeMaterialMask(MaterialMask, geom, prob_lo, prob_hi);
    InitializeMaterialMask(rFerroX, geom, MaterialMask);

    InitializeDopingProfiles(rFerroX, geom, AcceptorDoping, DonorDoping);
    
    if(Coordinate_Transformation == 1){
       Initialize_tphase_Mask(rFerroX, geom, tphaseMask);
       Initialize_Euler_angles(rFerroX, geom, angle_alpha, angle_beta, angle_theta);
    }

    bool contains_SC = false;

    FerroX_Util::Contains_sc(MaterialMask, contains_SC);
    //amrex::Print() << "contains_SC = " << contains_SC << "\n";

    std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2> LinOpBCType_2d;
    bool all_homogeneous_boundaries = true;
    bool some_functionbased_inhomogeneous_boundaries = false;
    bool some_constant_inhomogeneous_boundaries = false;

    SetPoissonBC(rFerroX, LinOpBCType_2d, all_homogeneous_boundaries, some_functionbased_inhomogeneous_boundaries, some_constant_inhomogeneous_boundaries);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    alpha_cc.setVal(0.);
    MultiFab beta_cc(ba, dm, 1, 1);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    // set cell-centered beta coefficient to permittivity based on mask
    InitializePermittivity(LinOpBCType_2d, beta_cc, MaterialMask, tphaseMask, n_cell, geom, prob_lo, prob_hi);
    eXstatic_MFab_Util::AverageCellCenteredMultiFabToCellFaces(beta_cc, beta_face);
    
    // INITIALIZE P in FE and rho in SC regions

    //InitializePandRho(P_old, Gamma, charge_den, e_den, hole_den, geom, prob_lo, prob_hi);//old
    InitializePandRho(P_old, Gamma, charge_den, e_den, hole_den, acceptor_den, donor_den, AcceptorDoping, DonorDoping, MaterialMask, tphaseMask, n_cell, geom, prob_lo, prob_hi);//mask based

    //for sundials 
    MultiFab::Copy(carrier_den, e_den, 0, 0, 1, 1);
    MultiFab::Copy(carrier_den, hole_den, 0, 1, 1, 1);

    //ComputePhi_Rho_Equilibrium(pMLMG, p_mlabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
    //               P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
    //               angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
    
    // time = starting time in the simulation
    Real time = 0.0;

    amrex::LPInfo info;
    std::unique_ptr<amrex::MLMG> pMLMG;
    std::unique_ptr<amrex::MLABecLaplacian> p_mlabec;
    int linop_maxorder = 2;
    int amrlev = 0; //refers to the setcoarsest level of the solve

    SetupMLMG(pMLMG, p_mlabec, LinOpBCType_2d, n_cell, beta_face, MaterialMask, acceptor_den, donor_den, rFerroX, PoissonPhi, time, info);

#ifdef AMREX_USE_EB
    std::unique_ptr<amrex::MLEBABecLap> p_mlebabec;
    SetupMLMG_EB(pMLMG, p_mlebabec, LinOpBCType_2d, n_cell, beta_face, MaterialMask, beta_cc, rFerroX, PoissonPhi, time, info);
#endif
    
    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int plt_step = 0;
        WritePlotfile(rFerroX, PoissonPhi, PoissonRHS, P_old, E, Jn, Jp, hole_den, e_den, acceptor_den, donor_den, charge_den, beta_cc, 
                      MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, Phidiff, geom, time, plt_step);
    }

    amrex::Print() << "=================== ADVANCE STEPS ===================== \n"<< std::endl;

    int steady_state_step = 1000000; //Initialize to a large number. It will be overwritten by the time step at which steady state condition is satidfied

    int sign = 1; //change sign to -1*sign whenever abs(Phi_Bc_hi) == Phi_Bc_hi_max to do triangular wave sweep
    int num_Vapp = 0;
    Real tiny = 1.e-6;    
 
#ifdef AMREX_USE_SUNDIALS

    TimeIntegrator<MultiFab> integrator(carrier_den);


    // use adaptive time step (dt used to set output times)
    bool adapt_dt = false;

    // adaptive time step relative and absolute tolerances
    Real reltol = 1.0e-4;
    Real abstol = 1.0e-9;

    ParmParse pp;
    // use adaptive step sizes
    pp.query("adapt_dt",adapt_dt);

    // adaptive step tolerances
    pp.query("reltol",reltol);
    pp.query("abstol",abstol);

#endif

    for (int step = 1; step <= nsteps; ++step)
    {
        Real step_strt_time = ParallelDescriptor::second();

	if (!use_sundials) {

#ifdef AMREX_USE_EB
            ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                              P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                              angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, prob_lo, prob_hi);
#else
            ComputePhi_Rho(pMLMG, p_mlabec, LinOpBCType_2d, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                           P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                           angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, time, prob_lo, prob_hi);
#endif

            // Calculate E from Phi
            ComputeEfromPhi(PoissonPhi, E, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

            // compute f^n = f(P^n,Phi^n)
            if (include_Landau == 1){
               Calculate_Landau(GL_rhs_Landau, P_old, Gamma, tphaseMask);
            }
            if (include_Grad == 1){
               Calculate_Grad(GL_rhs_grad, P_old, Gamma, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom);
            }
            if (include_Elec == 1){
               Calculate_Elec(GL_rhs_elec, E, Gamma, tphaseMask);
            }

            CalculateTDGL_RHS(GL_rhs, GL_rhs_Landau, GL_rhs_grad, GL_rhs_elec, P_old, E, Gamma, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom);

            // P^{n+1,*} = P^n + dt * f^n
            for (int i = 0; i < 3; i++){
                MultiFab::LinComb(P_new_pre[i], 1.0, P_old[i], 0, dt, GL_rhs[i], 0, 0, 1, Nghost);
                P_new_pre[i].FillBoundary(geom.periodicity()); 
            }
        	
            if (TimeIntegratorOrder == 1) {

                // copy new solution into old solution
                for (int i = 0; i < 3; i++){
                    MultiFab::Copy(P_old[i], P_new_pre[i], 0, 0, 1, 0);
                    // fill periodic ghost cells
                    P_old[i].FillBoundary(geom.periodicity());
                }

            } else {

#ifdef AMREX_USE_EB
                ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                                  P_new_pre, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                                  angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
                ComputePhi_Rho(pMLMG, p_mlabec, LinOpBCType_2d, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                               P_new_pre, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                               angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, time, prob_lo, prob_hi);
#endif

                //update E using PoissonPhi computed with P_new_pre
                ComputeEfromPhi(PoissonPhi, E, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

                // compute f^{n+1,*} = f(P^{n+1,*},Phi^{n+1,*})
                if (include_Landau == 1){
                   Calculate_Landau(GL_rhs_Landau, P_new_pre, Gamma, tphaseMask);
                }
                if (include_Grad == 1){
                   Calculate_Grad(GL_rhs_grad, P_new_pre, Gamma, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom);
                }
                if (include_Elec == 1){
                   Calculate_Elec(GL_rhs_elec, E, Gamma, tphaseMask);
                }

                CalculateTDGL_RHS(GL_rhs_pre, GL_rhs_Landau, GL_rhs_grad, GL_rhs_elec, P_new_pre, E, Gamma, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom);

                // P^{n+1} = P^n + dt/2 * f^n + dt/2 * f^{n+1,*}
                for (int i = 0; i < 3; i++){
                    MultiFab::LinComb(GL_rhs_avg[i], 0.5, GL_rhs[i], 0, 0.5, GL_rhs_pre[i], 0, 0, 1, Nghost);
                    MultiFab::LinComb(P_new[i], 1.0, P_old[i], 0, dt, GL_rhs_avg[i], 0, 0, 1, Nghost);
                }

                // copy new solution into old solution
                for (int i = 0; i < 3; i++){
                    MultiFab::Copy(P_old[i], P_new[i], 0, 0, 1, 0);
                    // fill periodic ghost cells
                    P_old[i].FillBoundary(geom.periodicity());
                }
            }
		
	} else { //using sundials

#ifdef AMREX_USE_SUNDIALS

#ifdef AMREX_USE_EB
            ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                          P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                          angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, prob_lo, prob_hi);
#else
            ComputePhi(pMLMG, p_mlabec, LinOpBCType_2d, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                       P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask,
                       angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, time, prob_lo, prob_hi);
#endif

            ComputeEfromPhi(PoissonPhi, E, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

            // Create a RHS source function we will integrate
            // for MRI this represents the slow processes
            auto rhs_fun = [&](MultiFab& rhs, const MultiFab& state, const Real& time) {
               BL_PROFILE_VAR("rhs_fun()", rhs_fun);

               rhs.setVal(0.);
	       MultiFab tmp_state(state, amrex::make_alias, 0, 2);
               tmp_state.FillBoundary(geom.periodicity());

               MultiFab e_den(state, amrex::make_alias, 0, 1);
               MultiFab p_den(state, amrex::make_alias, 1, 1);
               MultiFab e_rhs(rhs, amrex::make_alias, 0, 1);
               MultiFab p_rhs(rhs, amrex::make_alias, 1, 1);

               ComputeRHS_DriftDiffusion(PoissonPhi, e_rhs, p_rhs, Jn, Jp, e_den, p_den, MaterialMask, geom);
            };
 
            // Attach the right hand side and post-update functions
            // to the integrator
            integrator.set_rhs(rhs_fun);
     
 	    if (adapt_dt) {
                integrator.set_adaptive_step();
                integrator.set_tolerances(reltol, abstol);
            } else {
                integrator.set_time_step(dt);
            }

            // integrate forward one step from `time` by `dt`
            integrator.evolve(carrier_den, time+dt);
          
            // copy carrier_den comps into e_den, hole_den
            MultiFab::Copy(e_den, carrier_den, 0, 0, 1, 1);
            MultiFab::Copy(hole_den, carrier_den, 1, 0, 1, 1);
            
	    // Apply carrier BC and compute rho for the next time step
            ComputeRho_sundials(charge_den, e_den, hole_den, acceptor_den, donor_den, MaterialMask, geom);

#endif
	}

        // Check if steady state has reached 
    //    CheckSteadyState(PoissonPhi, PoissonPhi_Old, Phidiff, phi_tolerance, step, steady_state_step, inc_step); // Calculate E from Phi

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";
//        amrex::Print() << " \n";

        // update time
        time = time + dt;


        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && (step%plot_int == 0 || step == steady_state_step))
        {
            int plt_step = step;
            WritePlotfile(rFerroX, PoissonPhi, PoissonRHS, P_old, E, Jn, Jp, hole_den, e_den, acceptor_den, donor_den, charge_den, beta_cc, 
                      MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, Phidiff, geom, time, plt_step);
            
        }

        if(voltage_sweep == 1 && inc_step > 0 && step == inc_step)
        {
           //Update time-dependent Boundary Condition of Poisson's equation

            Phi_Bc_hi += sign*Phi_Bc_inc;
            num_Vapp += 1;
            if(std::abs(std::abs(Phi_Bc_hi) - Phi_Bc_hi_max) <= tiny) {
              sign *= -1;
              amrex::Print() << "Direction of voltage sweep is reversed. Phi_Bc_hi = " << Phi_Bc_hi << ", and Phi_Bc_hi_max = " << Phi_Bc_hi_max << std::endl;
            }
            amrex::Print() << "step = " << step << ", Phi_Bc_hi = " << Phi_Bc_hi << ", num_Vapp = " << num_Vapp << ", sign = " << sign << std::endl;

            // Set Dirichlet BC for Phi in z
            SetPhiBC_z(PoissonPhi, MaterialMask, n_cell, geom);
            //SetPhiBC_z(PoissonPhi, MaterialMask, acceptor_den, donor_den, n_cell, geom);

           // set Dirichlet BC by reading in the ghost cell values
#ifdef AMREX_USE_EB
           p_mlebabec->setLevelBC(amrlev, &PoissonPhi);

           if(rGprop.pEB->specify_inhomogeneous_dirichlet == 0)
           {
               p_mlebabec->setEBHomogDirichlet(amrlev, beta_cc);
           }
           else
           {
               p_mlebabec->setEBDirichlet(amrlev, *rGprop.pEB->p_surf_soln_union, beta_cc);
           }
#else 
           p_mlabec->setLevelBC(amrlev, &PoissonPhi);
#endif

#ifdef AMREX_USE_EB
           ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                   P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
           ComputePhi_Rho(pMLMG, p_mlabec, LinOpBCType_2d, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr,
                   P_old, charge_den, Jn, Jp, e_den, hole_den, acceptor_den, donor_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, n_cell, rFerroX, time, prob_lo, prob_hi);
#endif
        }//end inc_step	
   
        if (voltage_sweep == 0 && step == steady_state_step) {
           amrex::Print() << "voltage_sweep == 0 && step == steady_state_step!" << "\n";   
           break;
        }
        if (voltage_sweep == 1 && Phi_Bc_hi > 0. && Phi_Bc_hi - Phi_Bc_hi_max > tiny) {
           amrex::Print() << "voltage_sweep == 1 && Phi_Bc_hi > 0. && Phi_Bc_hi - Phi_Bc_hi_max > tiny!" << "\n";
           break;
        }        
        if (voltage_sweep == 1 && Phi_Bc_hi < 0. && -Phi_Bc_hi - Phi_Bc_hi_max > tiny) {
           amrex::Print() << "voltage_sweep == 1 && Phi_Bc_hi < 0. && -Phi_Bc_hi - Phi_Bc_hi_max > tiny!" << "\n";
           break;
        }   
        if (voltage_sweep == 1 && num_Vapp == num_Vapp_max) {
           amrex::Print() << "voltage_sweep == 1 && num_Vapp == num_Vapp_max!"  << "\n";
           break;
        }

    } // end step

    // MultiFab memory usage
    const int IOProc = ParallelDescriptor::IOProcessorNumber();

    amrex::Long min_fab_megabytes  = amrex::TotalBytesAllocatedInFabsHWM()/1048576;
    amrex::Long max_fab_megabytes  = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "High-water FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

    min_fab_megabytes  = amrex::TotalBytesAllocatedInFabs()/1048576;
    max_fab_megabytes  = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "Curent     FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";
    
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";

}
