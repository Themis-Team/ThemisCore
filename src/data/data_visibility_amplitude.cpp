/*!
  \file data_visibility_amplitude.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements datum and data objects containing visibility amplitudes.
  \details To be added
*/

#include "data_visibility_amplitude.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_visibility_amplitude::datum_visibility_amplitude(double up, double vp, double Vp, double errp, double frequencyp, double tp, std::string Station1p, std::string Station2p, std::string Sourcep)
  : V(Vp), err(errp), u(up), v(vp), 
    frequency(frequencyp), wavelength(constants::c/frequency), 
    tJ2000(tp),
    Station1(Station1p), Station2(Station2p),
    Source(Sourcep)
{
}

data_visibility_amplitude::data_visibility_amplitude()
  : _frequency(230e9)
{
  _visibilities.clear();
}

data_visibility_amplitude::data_visibility_amplitude(std::string file_name, const std::string time_type, bool read_frequency)
  : _frequency(230e9)
{
  _visibilities.clear();
  add_data(file_name, time_type, read_frequency);
}
  
  
data_visibility_amplitude::data_visibility_amplitude(std::vector<std::string> file_name, const std::vector<std::string> time_type)
  : _frequency(230e9)
{
  _visibilities.clear();
  if (time_type.empty())
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i],"HHMM");
  else
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i],time_type[i]);

}

data_visibility_amplitude::data_visibility_amplitude(std::string file_name, double frequency, const std::string time_type)
  : _frequency(frequency)
{
  _visibilities.clear();
  add_data(file_name, time_type, false, frequency);
}


void data_visibility_amplitude::add_data(std::string file_name, const std::string time_type, bool read_frequency, double frequency)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_visibility_amplitude::add_data: Can not open data file " << file_name << " for input. \n";
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
    double tmp_u, tmp_v, tmp_amp, tmp_err;
    double tmp_freq=_frequency;

    if (read_frequency) {
      ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_freq >> tmp_time >> tmp_base >> tmp_u >> tmp_v >> tmp_amp >> tmp_err;
      tmp_freq *= 1e9; // Supplied in GHz
    }
    else
    {
      ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_time >> tmp_base >> tmp_u >> tmp_v >> tmp_amp >> tmp_err;
      tmp_freq = frequency;
    }
    
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
      std::cerr << "data_visibility_amplitude: Invalid time format selected\n";
      std::exit(1);
    }
    double tJ2000 = utils::time_J2000(tmp_year,tmp_day,tmp_hour,tmp_min,tmp_sec);

    // In the standard format visibility amplitude data files,
    // u and v are given in Mega-lambda, convert these to units of lambda
    tmp_u *= 1e6;
    tmp_v *= 1e6;

    // Get the station codes for the two participating stations
    std::string tmp_S1, tmp_S2;
    if (tmp_base.size()==2) // Using one-letter station codes
    {
      tmp_S1 = tmp_base[0];
      tmp_S2 = tmp_base[1];
    }
    else if (tmp_base.size()==4) // Using two-letter station codes
    {
      tmp_S1 = tmp_base[0];
      tmp_S1 += tmp_base[1];
      tmp_S2 = tmp_base[2];
      tmp_S2 += tmp_base[3];
    }
    else // Unknown station codes, just copy
    {
      tmp_S1 = tmp_base;
      tmp_S2 = tmp_base;
    }
      
    _visibilities.push_back(new datum_visibility_amplitude(tmp_u,tmp_v,tmp_amp,tmp_err,tmp_freq,tJ2000,tmp_S1,tmp_S2,tmp_src));
  }
}

void data_visibility_amplitude::add_data(datum_visibility_amplitude& d)
{
  _visibilities.push_back(new datum_visibility_amplitude(d));
}
  
};

