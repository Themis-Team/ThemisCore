#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <mpi.h>

#include "model_image.h"
#include "model_image_multi_asymmetric_gaussian.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_multigaussian.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_crescent.h"
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
  Themis::model_image_multigaussian baseline_image(2);
  //Themis::model_image_multi_asymmetric_gaussian baseline_image(2);
  baseline_image.use_analytical_visibilities();
  
  //  Second, generate a summed model
  Themis::model_image_symmetric_gaussian sum_image1, sum_image2;
  //Themis::model_image_asymmetric_gaussian sum_image1, sum_image2;
  std::vector< Themis::model_image* > modelvec;
  modelvec.push_back(&sum_image1);
  modelvec.push_back(&sum_image2);
  //Themis::model_image_sum test_image(modelvec);
  Themis::model_image_sum test_image;
  test_image.add_model_image(sum_image1);
  test_image.add_model_image(sum_image2);
  
  Themis::model_image_crescent crescent;
  //test_image.add_model_image(crescent);


  sum_image1.use_analytical_visibilities();
  sum_image2.use_analytical_visibilities();
  crescent.use_analytical_visibilities();

  //sum_image1.use_numerical_visibilities();
  //sum_image2.use_numerical_visibilities();
  //crescent.use_numerical_visibilities();

  

  
  // Set a set of parameters
  double uas2rad = 1e-6/180./3600. * M_PI;
  std::vector<double> pmg;
  pmg.push_back(1.0); // I1
  pmg.push_back(25*uas2rad); // sigma1
  //pmg.push_back(0.5); // A1
  //pmg.push_back(0.0); // phi1
  pmg.push_back(0.0); // x1
  pmg.push_back(0.0); // y1
  pmg.push_back(0.5); // I2
  pmg.push_back(10*uas2rad); // sigma2
  //pmg.push_back(0.1); // A2
  //pmg.push_back(0.0); // phi2
  pmg.push_back(50.0*uas2rad); // x2
  pmg.push_back(-80.0*uas2rad); // y2
  pmg.push_back(0.0); // xi

  std::vector<double> pc;
  pc.push_back(2.0);
  pc.push_back(42*uas2rad);
  pc.push_back(0.3);
  pc.push_back(0.1);
  pc.push_back(1.0);

  std::vector<double> pis = pmg;
  pis[pmg.size()-1]=pc[0];
  pis.push_back(pc[1]);
  pis.push_back(pc[2]);
  pis.push_back(pc[3]);
  pis.push_back(pc[4]);


  baseline_image.generate_model(pmg);
  test_image.generate_model(pis);
  crescent.generate_model(pc);

  std::cerr << "Baseline size: " << baseline_image.size() << '\n';
  std::cerr << "Sum image size: " << test_image.size() << '\n';
  std::cerr << "Sum image1 image size: " << sum_image1.size() << '\n';
  std::cerr << "Sum image2 image size: " << sum_image2.size() << '\n';
  
  
  
  std::cout << "\n" << std::endl;

  // Run over list of points and print
  std::ofstream vcout("sum_test_vccomp.d");
  for (size_t i=0; i<dVMs.size(); ++i)
  {
    Themis::datum_visibility tmp(dVMs.datum(i).u,dVMs.datum(i).v,std::complex<double>(0,0),std::complex<double>(0,0),dVMs.datum(i).frequency,dVMs.datum(i).tJ2000,dVMs.datum(i).Station1,dVMs.datum(i).Station2,dVMs.datum(i).Source);


    std::complex<double> Vb = baseline_image.visibility(tmp,0);
    std::complex<double> Vt = test_image.visibility(tmp,0);
    std::complex<double> Vc = crescent.visibility(tmp,0);
    
    vcout << std::setw(15) << tmp.u/1.e9
	  << std::setw(15) << tmp.v/1.e9
	  << std::setw(15) << std::sqrt(tmp.u*tmp.u + tmp.v*tmp.v)/1e9
	  << std::setw(15) << Vb.real()/std::abs(Vb)
	  << std::setw(15) << Vb.imag()/std::abs(Vb)
	  << std::setw(15) << Vt.real()/std::abs(Vt)
	  << std::setw(15) << Vt.imag()/std::abs(Vt)
	  << std::setw(15) << Vc.real()/std::abs(Vc)
	  << std::setw(15) << Vc.imag()/std::abs(Vc)      
	  << std::endl;
  }

  
  // Run over list of points and print
  std::ofstream vmout("sum_test_vmcomp.d");
  for (size_t i=0; i<dVMs.size(); ++i)
  {
    vmout << std::setw(15) << dVMs.datum(i).u/1.e9
	  << std::setw(15) << dVMs.datum(i).v/1.e9
	  << std::setw(15) << std::sqrt(dVMs.datum(i).u*dVMs.datum(i).u + dVMs.datum(i).v*dVMs.datum(i).v)/1e9
	  << std::setw(15) << baseline_image.visibility_amplitude(dVMs.datum(i),0)
	  << std::setw(15) << test_image.visibility_amplitude(dVMs.datum(i),0)
	  << std::setw(15) << sum_image1.visibility_amplitude(dVMs.datum(i),0)
	  << std::endl;
  }

  
  std::ofstream cpout("sum_test_cpcomp.d");
  for (size_t i=0; i<dCPs.size(); ++i)
  {
    cpout << std::setw(15) << dCPs.datum(i).u1/1.e9
	  << std::setw(15) << dCPs.datum(i).v1/1.e9
	  << std::setw(15) << dCPs.datum(i).u2/1.e9
	  << std::setw(15) << dCPs.datum(i).v2/1.e9
	  << std::setw(15) << dCPs.datum(i).u3/1.e9
	  << std::setw(15) << dCPs.datum(i).v3/1.e9
	  << std::setw(15) << std::sqrt(dCPs.datum(i).u1*dCPs.datum(i).u1 + dCPs.datum(i).v1*dCPs.datum(i).v1+dCPs.datum(i).u2*dCPs.datum(i).u2 + dCPs.datum(i).v2*dCPs.datum(i).v2+dCPs.datum(i).u3*dCPs.datum(i).u3 + dCPs.datum(i).v3*dCPs.datum(i).v3)/1e9
	  << std::setw(15) << baseline_image.closure_phase(dCPs.datum(i),0)
	  << std::setw(15) << test_image.closure_phase(dCPs.datum(i),0)
	  << std::endl;
  }


  std::ofstream caout("sum_test_cacomp.d");
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
	  << std::setw(15) << baseline_image.closure_amplitude(dCAs.datum(i),0)
	  << std::setw(15) << test_image.closure_amplitude(dCAs.datum(i),0)
	  << std::endl;
  }


  //Finalize MPI
  MPI_Finalize();


  return 0;
}

