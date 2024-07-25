/*!
  \file data_closure_amplitude.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements datum and data objects containing closure amplitudes.
  \details To be added
*/

#include "data_closure_amplitude.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_closure_amplitude::datum_closure_amplitude(double u1p, double v1p, double u2p, double v2p, double u3p, double v3p, double CAp, double errp, double frequencyp, double tp, std::string Station1p, std::string Station2p, std::string Station3p, std::string Station4p, std::string Sourcep)
  : CA(CAp), err(errp), 
    u1(u1p), v1(v1p), u2(u2p), v2(v2p), u3(u3p), v3(v3p), u4(-u1p-u2p-u3p), v4(-v1p-v2p-v3p), 
    frequency(frequencyp), wavelength(constants::c/frequency), 
    tJ2000(tp),
    Station1(Station1p), Station2(Station2p), Station3(Station3p), Station4(Station4p),
    Source(Sourcep)
{
}

data_closure_amplitude::data_closure_amplitude()
{
  _closure_amplitudes.clear();
}

data_closure_amplitude::data_closure_amplitude(std::string file_name)
{
  _closure_amplitudes.clear();
  add_data(file_name);
}
  
data_closure_amplitude::data_closure_amplitude(std::vector<std::string> file_name)
{
  _closure_amplitudes.clear();
  for (size_t i=0; i<file_name.size(); ++i)
    add_data(file_name[i]);
}

void data_closure_amplitude::add_data(std::string file_name)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_closure_amplitude::add_data: Can not open datatype config file" << file_name << " for input. \n";
    std::exit(1);
  }
   
  while (ifile.good())
  {
    std::string str_line;
    getline (ifile, str_line);
	  
    if (str_line.empty()) {continue;}
    if (str_line[0] == '#') {continue;}
    
    std::stringstream ss_val;

    ss_val.str(std::string());
    ss_val.clear();
    ss_val << str_line;
    
    std::string tmp_src, tmp_quad;
    int tmp_year, tmp_day;
    double tmp_time;
    double tmp_u1, tmp_v1, tmp_u2, tmp_v2, tmp_u3, tmp_v3;
    double tmp_ca, tmp_err;

    ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_time >> tmp_quad >> tmp_u1 >> tmp_v1 >> tmp_u2 >> tmp_v2 >> tmp_u3 >> tmp_v3 >> tmp_ca >> tmp_err;

    int tmp_hour, tmp_min;
    double tmp_sec;

    tmp_hour = int(tmp_time);
    tmp_min = int((tmp_time-double(tmp_hour))*60);
    tmp_sec = (tmp_time-double(tmp_hour)-double(tmp_min)/60.0)*3600.0;
    


    double tJ2000 = utils::time_J2000(tmp_year,tmp_day,tmp_hour,tmp_min,tmp_sec);

    // In the standard format visibility amplitude data files,
    // u and v are given in Mega-lambda, conver these to units of lambda
    tmp_u1 *= 1e6;
    tmp_v1 *= 1e6;
    tmp_u2 *= 1e6;
    tmp_v2 *= 1e6;
    tmp_u3 *= 1e6;
    tmp_v3 *= 1e6;


    // Get the station codes for the two participating stations
    std::string tmp_S1, tmp_S2, tmp_S3, tmp_S4;
    if (tmp_quad.size()==4) // Using one-letter station codes
    {
      tmp_S1 = tmp_quad[0];
      tmp_S2 = tmp_quad[1];
      tmp_S3 = tmp_quad[2];
      tmp_S4 = tmp_quad[3];
    }
    else if (tmp_quad.size()==8) // Using two-letter station codes
    {
      tmp_S1 = tmp_quad[0];
      tmp_S1 += tmp_quad[1];
      tmp_S2 = tmp_quad[2];
      tmp_S2 += tmp_quad[3];
      tmp_S3 = tmp_quad[4];
      tmp_S3 += tmp_quad[5];
      tmp_S4 = tmp_quad[6];
      tmp_S4 += tmp_quad[7];
    }
    else // Unknown station codes, just copy
    {
      tmp_S1 = tmp_quad;
      tmp_S2 = tmp_quad;
      tmp_S3 = tmp_quad;
      tmp_S4 = tmp_quad;
    }


    
    _closure_amplitudes.push_back(new datum_closure_amplitude(tmp_u1,tmp_v1,tmp_u2,tmp_v2,tmp_u3,tmp_v3,tmp_ca,tmp_err,230e9,tJ2000,tmp_S1,tmp_S2,tmp_S3,tmp_S4,tmp_src));
  }
}

void data_closure_amplitude::add_data(datum_closure_amplitude& d)
{
  _closure_amplitudes.push_back(new datum_closure_amplitude(d));
}
							    
};
