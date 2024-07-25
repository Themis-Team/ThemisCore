#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <mpi.h>

#include "model_image_xsringauss.h"
#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_closure_amplitude.h"
#include "data_visibility.h"
#include "utils.h"

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in some data to get some baselines
  Themis::data_visibility_amplitude dVMs(Themis::utils::global_path("sim_data/Challenge3/challenge03a_vtable.d"));
 Themis::data_closure_phase dCPs(Themis::utils::global_path("sim_data/Challenge3/challenge03b_btable.wo_trivial.d"));
  Themis::data_closure_amplitude dCAs(Themis::utils::global_path("sim_data/Challenge3/challenge03b_ctable.wo_trivial.d"));


  // Define two sets of models to compare to each other:
  //  First the multi-Gaussian
  Themis::model_image_xsringauss xrg_n, xrg_a;
  xrg_n.use_numerical_visibilities();
  xrg_a.use_analytical_visibilities();
  
  
  // Set a set of parameters
  double uas2rad = 1e-6/180./3600. * M_PI;
  std::vector<double> p;
  p.push_back(1.0); // Itot
  p.push_back(25*uas2rad); // Rp
  p.push_back(0.3); // psi
  p.push_back(0.9); // tau
  p.push_back(0.5); // f
  p.push_back(0.5); // sig_g
  p.push_back(0.5); // axis ratio
  p.push_back(0.3); // flux ratio
  p.push_back(0.0); // flux ratio

  xrg_n.generate_model(p);
  xrg_a.generate_model(p);
  
  // Run over list of points and print
  std::ofstream vcout("xsringauss_test_vccomp.d");
  for (size_t i=0; i<dVMs.size(); ++i)
  {
    Themis::datum_visibility tmp(dVMs.datum(i).u,dVMs.datum(i).v,std::complex<double>(0,0),std::complex<double>(0,0),dVMs.datum(i).frequency,dVMs.datum(i).tJ2000,dVMs.datum(i).Station1,dVMs.datum(i).Station2,dVMs.datum(i).Source);


    std::complex<double> Vb = xrg_n.visibility(tmp,0);
    std::complex<double> Vt = xrg_a.visibility(tmp,0);

    vcout << std::setw(15) << tmp.u/1.e9
	  << std::setw(15) << tmp.v/1.e9
	  << std::setw(15) << std::sqrt(tmp.u*tmp.u + tmp.v*tmp.v)/1e9
	  << std::setw(15) << Vb.real()/std::abs(Vb)
	  << std::setw(15) << Vb.imag()/std::abs(Vb)
	  << std::setw(15) << Vt.real()/std::abs(Vt)
	  << std::setw(15) << Vt.imag()/std::abs(Vt)
	  << std::endl;
  }

  
  // Run over list of points and print
  std::ofstream vmout("xsringauss_test_vmcomp.d");
  for (size_t i=0; i<dVMs.size(); ++i)
  {
    vmout << std::setw(15) << dVMs.datum(i).u/1.e9
	  << std::setw(15) << dVMs.datum(i).v/1.e9
	  << std::setw(15) << std::sqrt(dVMs.datum(i).u*dVMs.datum(i).u + dVMs.datum(i).v*dVMs.datum(i).v)/1e9
	  << std::setw(15) << xrg_n.visibility_amplitude(dVMs.datum(i),0)
	  << std::setw(15) << xrg_a.visibility_amplitude(dVMs.datum(i),0)
	  << std::endl;
  }

  
  std::ofstream cpout("xsringauss_test_cpcomp.d");
  for (size_t i=0; i<dCPs.size(); ++i)
  {
    cpout << std::setw(15) << dCPs.datum(i).u1/1.e9
	  << std::setw(15) << dCPs.datum(i).v1/1.e9
	  << std::setw(15) << dCPs.datum(i).u2/1.e9
	  << std::setw(15) << dCPs.datum(i).v2/1.e9
	  << std::setw(15) << dCPs.datum(i).u3/1.e9
	  << std::setw(15) << dCPs.datum(i).v3/1.e9
	  << std::setw(15) << std::sqrt(dCPs.datum(i).u1*dCPs.datum(i).u1 + dCPs.datum(i).v1*dCPs.datum(i).v1+dCPs.datum(i).u2*dCPs.datum(i).u2 + dCPs.datum(i).v2*dCPs.datum(i).v2+dCPs.datum(i).u3*dCPs.datum(i).u3 + dCPs.datum(i).v3*dCPs.datum(i).v3)/1e9
	  << std::setw(15) << xrg_n.closure_phase(dCPs.datum(i),0)
	  << std::setw(15) << xrg_a.closure_phase(dCPs.datum(i),0)
	  << std::endl;
  }


  std::ofstream caout("xsringauss_test_cacomp.d");
  for (size_t i=0; i<dCAs.size(); ++i)
  {
    caout << std::setw(15) << dCAs.datum(i).u1/1.e9
	  << std::setw(15) << dCAs.datum(i).v1/1.e9
	  << std::setw(15) << dCAs.datum(i).u2/1.e9
	  << std::setw(15) << dCAs.datum(i).v2/1.e9
	  << std::setw(15) << dCAs.datum(i).u3/1.e9
	  << std::setw(15) << dCAs.datum(i).v3/1.e9
	  << std::setw(15) << dCAs.datum(i).u4/1.e9
	  << std::setw(15) << dCAs.datum(i).v4/1.e9
	  << std::setw(15) << std::sqrt(dCAs.datum(i).u1*dCAs.datum(i).u1 + dCAs.datum(i).v1*dCAs.datum(i).v1+dCAs.datum(i).u2*dCAs.datum(i).u2 + dCAs.datum(i).v2*dCAs.datum(i).v2+dCAs.datum(i).u3*dCAs.datum(i).u3 + dCAs.datum(i).v3*dCAs.datum(i).v3+dCAs.datum(i).u4*dCAs.datum(i).u4 + dCAs.datum(i).v4*dCAs.datum(i).v4)/1e9
	  << std::setw(15) << xrg_n.closure_amplitude(dCAs.datum(i),0)
	  << std::setw(15) << xrg_a.closure_amplitude(dCAs.datum(i),0)
	  << std::endl;
  }


  std::ofstream imout("xsringauss_test_image.d");
  // Access the image data (not usually required)
  std::vector<std::vector<double> > alpha, beta, I;
  xrg_n.get_image(alpha,beta,I);

  // Convert from radians to uas
  double rad2M = 1.0/uas2rad;
  imout << "Nx:"
	<< std::setw(15) << rad2M*alpha[0][0]
	<< std::setw(15) << rad2M*alpha[alpha.size()-1].back()
	<< std::setw(15) << alpha.size()
	<< '\n';
  imout << "Ny:"
	<< std::setw(15) << rad2M*beta[0][0]
	<< std::setw(15) << rad2M*beta[beta.size()-1].back()
	<< std::setw(15) << beta.size()
	<< '\n';
  imout << std::setw(5) << "i"
	<< std::setw(5) << "j"
	<< std::setw(15) << "I (Jy/sr)"
	<< '\n';
  for (size_t j=0; j<alpha[0].size(); ++j)
    for (size_t i=0; i<alpha.size(); ++i)
      imout << std::setw(5) << i
	    << std::setw(5) << j
	    << std::setw(15) << I[i][j]
	    << '\n';
  


  //Finalize MPI
  MPI_Finalize();


  return 0;
}

