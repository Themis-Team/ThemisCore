/*!
  \file data_crosshand_visibilities.cpp
  \author Avery E. Broderick
  \date  March, 2020
  \brief Implements datum and data objects containing crosshand visibilities (RR, LL, RL, LR).
  \details To be added
*/

#include "data_crosshand_visibilities.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_crosshand_visibilities::datum_crosshand_visibilities(double up, double vp, double phi1p, double phi2p, std::complex<double> RRp, std::complex<double> RRerrp, std::complex<double> LLp, std::complex<double> LLerrp, std::complex<double> RLp, std::complex<double> RLerrp, std::complex<double> LRp, std::complex<double> LRerrp, double frequencyp, double tp, std::string Station1p, std::string Station2p, std::string Sourcep)
  : RR(RRp), LL(LLp), RL(RLp), LR(LRp), RRerr(RRerrp), LLerr(LLerrp), RLerr(RLerrp), LRerr(LRerrp), u(up), v(vp), phi1(phi1p), phi2(phi2p),
    frequency(frequencyp), wavelength(constants::c/frequency), 
    tJ2000(tp),
    Station1(Station1p), Station2(Station2p),
    Source(Sourcep)
{
}

data_crosshand_visibilities::data_crosshand_visibilities()
{
  _visibilities.clear();
}

data_crosshand_visibilities::data_crosshand_visibilities(std::string file_name, const std::string time_type, bool read_frequency)
  : _frequency(230e9)
{
  _visibilities.clear();
  add_data(file_name, time_type, read_frequency);
}
  
data_crosshand_visibilities::data_crosshand_visibilities(std::vector<std::string> file_name, const std::vector<std::string> time_type)
  : _frequency(230e9)
{
  _visibilities.clear();
  if (time_type.empty())
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i]);
  else
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i], time_type[i]);

}

void data_crosshand_visibilities::add_data(std::string file_name, const std::string time_type, bool read_frequency)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_crosshand_visibilities::add_data: Can not open data file " << file_name << " for input. \n";
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
    double tmp_u, tmp_v;
    double tmp_phi1, tmp_phi2;
    double tmp_RRr, tmp_RRi, tmp_RRerrr, tmp_RRerri;
    double tmp_LLr, tmp_LLi, tmp_LLerrr, tmp_LLerri;
    double tmp_RLr, tmp_RLi, tmp_RLerrr, tmp_RLerri;
    double tmp_LRr, tmp_LRi, tmp_LRerrr, tmp_LRerri;
    double tmp_freq=_frequency;

    ss_val >> tmp_src >> tmp_year >> tmp_day;
    if (read_frequency) {
      ss_val >> tmp_freq;
      tmp_freq *= 1e9; // Supplied in GHz
    }
    ss_val >> tmp_time >> tmp_base >> tmp_u >> tmp_v;
    ss_val >> tmp_phi1 >> tmp_phi2;
    ss_val >> tmp_RRr >> tmp_RRerrr >> tmp_RRi >> tmp_RRerri;
    ss_val >> tmp_LLr >> tmp_LLerrr >> tmp_LLi >> tmp_LLerri;
    ss_val >> tmp_RLr >> tmp_RLerrr >> tmp_RLi >> tmp_RLerri;
    ss_val >> tmp_LRr >> tmp_LRerrr >> tmp_LRi >> tmp_LRerri;

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
      std::cerr << "data_crosshand_visibilities: Invalid time format selected\n";
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
      
    //_visibilities.push_back(new datum_crosshand_visibilities(tmp_u,tmp_v,std::complex<double>(tmp_Vr,tmp_Vi),std::complex<double>(tmp_errr,tmp_erri),tmp_freq,tJ2000,tmp_S1,tmp_S2,tmp_src));
    _visibilities.push_back(new datum_crosshand_visibilities(tmp_u,tmp_v,tmp_phi1,tmp_phi2,
							     std::complex<double>(tmp_RRr,tmp_RRi),std::complex<double>(tmp_RRerrr,tmp_RRerri),
							     std::complex<double>(tmp_LLr,tmp_LLi),std::complex<double>(tmp_LLerrr,tmp_LLerri),
							     std::complex<double>(tmp_RLr,tmp_RLi),std::complex<double>(tmp_RLerrr,tmp_RLerri),
							     std::complex<double>(tmp_LRr,tmp_LRi),std::complex<double>(tmp_LRerrr,tmp_LRerri),
							     tmp_freq,tJ2000,tmp_S1,tmp_S2,tmp_src));
  }
}

void data_crosshand_visibilities::add_data(datum_crosshand_visibilities& d)
{
  _visibilities.push_back(new datum_crosshand_visibilities(d));
}
  
};

