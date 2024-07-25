/*! 
  \file likelihood_marginalized_closure_phase.cpp
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Implementation file for the Marginalized Closure Phase Likelihood class
*/


#include "likelihood_marginalized_closure_phase.h"
#include "data_closure_phase.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_marginalized_closure_phase::likelihood_marginalized_closure_phase(
  data_closure_phase& data, model_closure_phase& model, double sigma_phi)
    : _data(data), _model(model), _sigma_phi(sigma_phi), _phiMarg(0.0), _phiMax(0.0)  //local variables
  {
  }

  void likelihood_marginalized_closure_phase::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }
    
  double likelihood_marginalized_closure_phase::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
      
    double phiM, phiM_num, phiM_den;
    double DeltaPhi;
    double LMax, LMarg;
    double Sigma;
      
    phiM = phiM_num = phiM_den = 0.0;

    for(size_t i = 0; i < _data.size(); ++i)
    {
      DeltaPhi = angle_difference(_data.datum(i).CP,_model.closure_phase(_data.datum(i),0.25*_data.datum(i).err));
      
      phiM_num += DeltaPhi/(_data.datum(i).err*_data.datum(i).err);

      phiM_den += 1.0 / (_data.datum(i).err*_data.datum(i).err);
    }
      
    //phi that maximizes the likelihood
    phiM = phiM_num/phiM_den;
          
    LMax = 0.0;
      
    for (size_t i=0; i < _data.size(); ++i)    
    {
      DeltaPhi = angle_difference(_data.datum(i).CP,_model.closure_phase(_data.datum(i),0.25*_data.datum(i).err)+phiM);
      
      LMax += -0.5*std::pow(DeltaPhi/_data.datum(i).err, 2);
    }

    // Following Avery, et.al. (2016), we consider a Gaussian prior for phi with a width sigma_phi
    Sigma = 1/std::sqrt(phiM_den);
    
    // Marginalized value of phi
    _phiMarg = _sigma_phi*_sigma_phi*phiM / (_sigma_phi*_sigma_phi + Sigma*Sigma); 

    // Value of phi that maximizes the likelihood
    _phiMax = phiM;
    
    //Marginalized Likelihood
    LMarg = LMax + std::log(Sigma/std::sqrt(_sigma_phi*_sigma_phi + Sigma*Sigma))
            - 0.5*phiM*phiM/(_sigma_phi*_sigma_phi + Sigma*Sigma);


    return LMarg;
  }


  double likelihood_marginalized_closure_phase::chi_squared(std::vector<double>& x)
  {
    
    _model.generate_model(x);
      
    double phiM, phiM_num, phiM_den;
    double DeltaPhi;
    double LMax;
      
    phiM = phiM_num = phiM_den = 0.0;

    for(size_t i = 0; i < _data.size(); ++i)
    {
      DeltaPhi = angle_difference(_data.datum(i).CP,_model.closure_phase(_data.datum(i),0.25*_data.datum(i).err));

      phiM_num += DeltaPhi/(_data.datum(i).err*_data.datum(i).err);

      //phiM_num += ( _data.datum(i).CP
      //	    - _model.closure_phase(_data.datum(i),0.25*_data.datum(i).err) )
      //          /(_data.datum(i).err*_data.datum(i).err);

      phiM_den += 1.0 / (_data.datum(i).err*_data.datum(i).err);
    }
      
    //phi that maximizes the likelihood
    phiM = phiM_num/phiM_den;
          
    LMax = 0.0;
      
    for (size_t i=0; i < _data.size(); ++i)    
    {
      DeltaPhi = angle_difference(_data.datum(i).CP,_model.closure_phase(_data.datum(i),0.25*_data.datum(i).err)+phiM);
      
      LMax += -0.5*std::pow(DeltaPhi/_data.datum(i).err, 2);
    }

    //std::cout << "Marginalized closure phase Chi2: " << (-2.0*LMax) << std::endl;                  
    return (-2.0*LMax);
  }


  double likelihood_marginalized_closure_phase::get_marginalized_phi() const
  {
    return ( _phiMarg );
  }

  double likelihood_marginalized_closure_phase::get_maximizing_phi() const
  {
    return ( _phiMax );
  }

  double likelihood_marginalized_closure_phase::angle_difference(double a, double b) const
  {
    double dab = (M_PI/180.0)*(a-b);
    return ( (180.0/M_PI)*std::atan2( std::sin(dab), std::cos(dab) ) );
  }

  void likelihood_marginalized_closure_phase::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_marginalized_closure_phase output file\n#"
	  << std::setw(14) << "u1 (Gl)"
	  << std::setw(15) << "v1 (Gl)"
	  << std::setw(15) << "u2 (Gl)"
	  << std::setw(15) << "v2 (Gl)"
	  << std::setw(15) << "u3 (Gl)"
	  << std::setw(15) << "v3 (Gl)"
	  << std::setw(15) << "CP (deg)"
	  << std::setw(15) << "err (deg)"
	  << std::setw(15) << "model CP (deg)"
	  << std::setw(15) << "residual (deg)"
	  << '\n';


    double CP, CPsig, CP0;
    for (size_t i=0; i<_data.size(); ++i)
    {
      CP = _data.datum(i).CP;
      CPsig = _data.datum(i).err;
      CP0 = _model.closure_phase(_data.datum(i),0.25*CPsig)+_phiMax;
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u1/1e9
	    << std::setw(15) << _data.datum(i).v1/1e9
	    << std::setw(15) << _data.datum(i).u2/1e9
	    << std::setw(15) << _data.datum(i).v2/1e9
	    << std::setw(15) << _data.datum(i).u3/1e9
	    << std::setw(15) << _data.datum(i).v3/1e9
	    << std::setw(15) << CP
	    << std::setw(15) << CPsig
	    << std::setw(15) << CP0
	    << std::setw(15) << angle_difference(CP,CP0)
	    << '\n';
    }
  }
};
