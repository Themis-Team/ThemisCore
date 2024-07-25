/*! 
  \file likelihood_marginalized_visibility_amplitude.cpp
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Implementation file for the Marginalized Visibility Amplitude Likelihood class
*/


#include "likelihood_marginalized_visibility_amplitude.h"
#include <cmath>

#include <iostream>
#include <iomanip>

namespace Themis
{

  likelihood_marginalized_visibility_amplitude::likelihood_marginalized_visibility_amplitude(
  data_visibility_amplitude& data, model_visibility_amplitude& model)
    : _data(data), _model(model)
  {
  }

  void likelihood_marginalized_visibility_amplitude::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  double likelihood_marginalized_visibility_amplitude::operator()(std::vector<double>& x)
  {
    
    /*std::cout << "likelihood_marginalized_visibility_amplitude::operator()"
	      << std::setw(5) << MPI::COMM_WORLD.Get_rank()
	      << std::setw(15) << x[0]
	      << std::setw(15) << x[1]
	      << std::setw(15) << x[2]
	      << std::endl;
    */

    
    _model.generate_model(x);
    
    double V00;
    double yi, yi_hat;
    double V00M_num, V00M_den;
    double LMax, LMarg;
    
    // Getting V00, the visility at (u,v) = (0,0)
    datum_visibility_amplitude d(0.0,0.0,1.0,0.1);
    V00 = _model.visibility_amplitude(d,0);
    //std::cout << "V00: " << V00 << std::endl;
    
    //Initializing V00M
    V00M_den = V00M_num = 0.0;
    
    //Calculating V00M_num and V00M_den
    for(size_t i = 0; i < _data.size(); ++i)
    {
      yi = _data.datum(i).V / _data.datum(i).err;
      yi_hat = _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)
	       /(V00 * _data.datum(i).err);
      //std::cout << "Y_hat: " << V00 << std::endl;
      V00M_num += yi*yi_hat;
      V00M_den += yi_hat*yi_hat;
    }
    
    _V00M = V00M_num/V00M_den;

    //Initializing LM for the maximized likelihood
    LMax = 0.0;

    for (size_t i=0; i < _data.size(); ++i)    
    {
      //"normalize" theoretical visibility by / V00 and * V00M
      LMax += -0.5*std::pow((_data.datum(i).V - _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)*_V00M/V00)/_data.datum(i).err, 2);
    }
    
    //The marginalized Likelihood assuming a flat prior on V00 is then:
    LMarg = LMax + 0.5*std::log(2*M_PI*V00*V00/V00M_den);    

    return LMarg;
  }


  double likelihood_marginalized_visibility_amplitude::chi_squared(std::vector<double>& x)
  {
    _model.generate_model(x);

    double V00;
    double yi, yi_hat;
    double V00M, V00M_num, V00M_den;
    double LMax;

    // Getting V00, the visility at (u,v) = (0,0)
    datum_visibility_amplitude d(0.0,0.0,1.0,0.1);
    V00 = _model.visibility_amplitude(d,0);

    //Initializing V00M
    V00M = V00M_den = V00M_num = 0;

    //Calculating V00M_num and V00M_den
    for(size_t i = 0; i < _data.size(); ++i)
    {
      yi = _data.datum(i).V / _data.datum(i).err;
      yi_hat = _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)
                /(V00 * _data.datum(i).err);
      V00M_num += yi*yi_hat;
      V00M_den += yi_hat*yi_hat;
    }

    V00M = V00M_num/V00M_den;

    //Initializing LM for the maximized likelihood
    LMax = 0.0;

    for (size_t i=0; i < _data.size(); ++i)    
    {
      //"normalize" theoretical visibility by / V00 and * V00M
      LMax += -0.5*std::pow((_data.datum(i).V - _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)*V00M/V00)/_data.datum(i).err, 2);
    }

    //std::cout << "Marginalized visibiliy Chi2: " << (-2.0*LMax) << std::endl;    
    return (-2.0*LMax);

  }

  double likelihood_marginalized_visibility_amplitude::get_disk_intensity_normalization()
  {
    return ( _V00M );
  }

  void likelihood_marginalized_visibility_amplitude::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_marginalized_visibility_amplitude output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "|V| (Jy)"
	  << std::setw(15) << "err (Jy)"
	  << std::setw(15) << "model |V| (Jy)"
	  << std::setw(15) << "residual (Jy)"
	  << '\n';


    datum_visibility_amplitude d(0.0,0.0,1.0,0.1);
    double V00 = _model.visibility_amplitude(d,0);
    
    for (size_t i=0; i<_data.size(); ++i)
    {
      double V = (_V00M/V00) * _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err);
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).V
	    << std::setw(15) << _data.datum(i).err
	    << std::setw(15) << V
	    << std::setw(15) << (_data.datum(i).V-V)
	    << '\n';
    }
  }
  
};
