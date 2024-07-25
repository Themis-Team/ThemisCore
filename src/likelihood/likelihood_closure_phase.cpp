/*!
  \file likelihood_closure_phase.cpp
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Implementation file for the Closure Phase Likelihood class
  \details
*/

#include "likelihood_closure_phase.h"
#include <math.h>
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_closure_phase::likelihood_closure_phase(data_closure_phase& data, model_closure_phase& model)
    : _data(data), _model(model)
  {
  }
  
  void likelihood_closure_phase::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }
  
  double likelihood_closure_phase::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
    double sum = 0.0;
    double DeltaPhi;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      DeltaPhi = angle_difference(_data.datum(i).CP,_model.closure_phase(_data.datum(i),0.25*_data.datum(i).err));
      
      sum += - 0.5*std::pow(DeltaPhi/_data.datum(i).err,2);
    }
    
    return sum;
  }
  
  double likelihood_closure_phase::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }
  
  double likelihood_closure_phase::angle_difference(double a, double b) const
  {
    double dab = (M_PI/180.0)*(a-b);
    return ( (180.0/M_PI)*std::atan2( std::sin(dab), std::cos(dab) ) );
  }

  
  void likelihood_closure_phase::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_closure_phase output file\n#"
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
      CP0 = _model.closure_phase(_data.datum(i),0.25*CPsig);
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
