/*!
  \file model_image_fixed_lightcurve.cpp
  \author Avery Broderick
  \date  September, 2020
  \brief Header file for the model_image_fixed_lightcurve class, which renormalizes the visibilities by a fixed time-variable normalization.  The provides a way to incorporate flux variability that does not impact image structure.
  \details To be added
*/

#include "model_image_fixed_lightcurve.h"
#include "data_visibility.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace Themis {

  model_image_fixed_lightcurve::model_image_fixed_lightcurve(model_image& image, std::vector<double> t, std::vector<double> F)
    : _image(image), _light_curve_table("mcubic")
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_fixed_lightcurve in rank " << world_rank << std::endl;

    set_light_curve(t,F);
  }

  model_image_fixed_lightcurve::model_image_fixed_lightcurve(model_image& image, std::string lightcurve_filename, std::string time_type)
    : _image(image), _light_curve_table("mcubic")
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_fixed_lightcurve in rank " << world_rank << std::endl;

    set_light_curve(lightcurve_filename,time_type);
  }

  void model_image_fixed_lightcurve::set_light_curve(std::vector<double> t, std::vector<double> F)
  {
    _light_curve_table.set_tables(t,F);

    _average_flux = 0.0;
    for (size_t i=0; i<F.size(); ++i)
      _average_flux += F[i];
    _average_flux /= double(F.size());
  }

  
  void model_image_fixed_lightcurve::set_light_curve(std::string lightcurve_filename, std::string time_type)
  {
    std::vector<double> t,F;
    std::ifstream ifile;
    ifile.open(lightcurve_filename.c_str());
    if(!ifile.is_open())
    {
      std::cerr << "data_visibility::add_data: Can not open data file " << lightcurve_filename << " for input. \n";
      std::exit(1);
    }
   
    while (ifile.good())
    {
      std::string str_line;
      std::getline (ifile, str_line);
      
      if (str_line.empty()) {continue;}
      if (str_line[0] == '#') {continue;}
      
      std::stringstream ss_val;
      
      ss_val.str(std::string());
      ss_val.clear();
      ss_val << str_line;

      std::string tmp_src;
      int tmp_year, tmp_day;
      double tmp_time, tmp_flux;
      
      ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_time >> tmp_flux;
      
      int tmp_hour, tmp_min;
      double tmp_sec;
      if ( time_type == "HH")
      {
	tmp_hour = int(tmp_time);
	tmp_min = int((tmp_time-double(tmp_hour))*60);
	tmp_sec = (tmp_time-double(tmp_hour)-double(tmp_min)/60.0)*3600.0;
      }
      else if ( time_type == "HHMM")
      {
	tmp_hour = int(tmp_time/100.0);
	tmp_min = int(tmp_time-tmp_hour*100);
	tmp_sec = tmp_time-double(tmp_hour)*100-double(tmp_min);
      }
      else
      {
	std::cerr << "data_visibility: Invalid time format selected\n";
	std::exit(1);
      }
      double tJ2000 = utils::time_J2000(tmp_year,tmp_day,tmp_hour,tmp_min,tmp_sec);
      
      t.push_back(tJ2000);
      F.push_back(tmp_flux);
    }
    _light_curve_table.set_tables(t,F);

    _average_flux = 0.0;
    for (size_t i=0; i<F.size(); ++i)
      _average_flux += F[i];
    _average_flux /= double(F.size());
  }

  void model_image_fixed_lightcurve::generate_model(std::vector<double> parameters)
  {
    _image.generate_model(parameters);
  }
 

  void model_image_fixed_lightcurve::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    _image.get_image(alpha,beta,I);
    for (size_t i=0; i<I.size(); ++i)
      for (size_t j=0; j<I[i].size(); ++j)
	I[i][j] *= _average_flux;
  } 


  std::complex<double> model_image_fixed_lightcurve::visibility(datum_visibility& d, double acc)
  {
    return ( _image.visibility(d,acc) * _light_curve_table(d.tJ2000) );
  }

  double model_image_fixed_lightcurve::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    return ( _image.visibility_amplitude(d,acc) * _light_curve_table(d.tJ2000) );
  }

  double model_image_fixed_lightcurve::closure_phase(datum_closure_phase& d, double acc)
  {
    return ( _image.closure_phase(d,acc) );
  }

  double model_image_fixed_lightcurve::closure_amplitude(datum_closure_amplitude& d, double acc)
  { 
    return ( _image.closure_amplitude(d,acc) );
  }

  void model_image_fixed_lightcurve::set_mpi_communicator(MPI_Comm comm)
  {
      _image.set_mpi_communicator(comm);
  }
  
};
