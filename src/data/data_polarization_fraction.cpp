/*!
  \file data_polarization_fraction.cpp
  \author Roman Gold
  \date Apr 2017
  \brief Implements a fractional polarization data class.  
  \details Collections of mbreve data are defined in data_polarization_fraction, which includes simple I/O tools and provides access to a list of appropriately constructed datum_mbreve objects.
*/

#include "data_polarization_fraction.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_polarization_fraction::datum_polarization_fraction(double up, double vp, double mbrevep_amp, double errp, double frequencyp, double tp, std::string Station1p, std::string Station2p, std::string Sourcep)
  : mbreve_amp(mbrevep_amp), err(errp), u(up), v(vp), 
    frequency(frequencyp), wavelength(constants::c/frequency), 
    tJ2000(tp),
    Station1(Station1p), Station2(Station2p),
    Source(Sourcep)
{
}

data_polarization_fraction::data_polarization_fraction()
{
  _mbreve.clear();
}

data_polarization_fraction::data_polarization_fraction(std::string file_name)
{
  _mbreve.clear();
  add_data(file_name);
}
  
data_polarization_fraction::data_polarization_fraction(std::vector<std::string> file_name)
{
  _mbreve.clear();
  for (size_t i=0; i<file_name.size(); ++i)
    add_data(file_name[i]);
}

void data_polarization_fraction::add_data(std::string file_name)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_polarization_fraction::add_data: Can not open data file" << file_name << " for input. \n";
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
    
    std::string tmp_src, tmp_base;
    int tmp_year, tmp_day;
    double tmp_time;
    double tmp_u, tmp_v, tmp_mbreve_amp, tmp_mbreve_amp_err;

    ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_time >> tmp_base >> tmp_u >> tmp_v >> tmp_mbreve_amp >> tmp_mbreve_amp_err;
    
    int tmp_hour, tmp_min;
    double tmp_sec;

    tmp_hour = int(tmp_time);
    tmp_min = int((tmp_time-double(tmp_hour))*60);
    tmp_sec = 0.; // :-s
    // tmp_sec = tmp_time-double(tmp_hour)*100-double(tmp_min);
    
    double tJ2000 = utils::time_J2000(tmp_year,tmp_day,tmp_hour,tmp_min,tmp_sec);

    // In the standard format visibility amplitude data files,
    // u and v are given in Mega-lambda, conver these to units of lambda
    tmp_u *= 1e6;
    tmp_v *= 1e6;

    _mbreve.push_back(new datum_polarization_fraction(tmp_u,tmp_v,tmp_mbreve_amp,tmp_mbreve_amp_err,230e9,tJ2000,tmp_base,tmp_base,tmp_src));
  }
}

void data_polarization_fraction::add_data(datum_polarization_fraction& d)
{
  _mbreve.push_back(new datum_polarization_fraction(d));
}
  
};

