/*!
  \file data_closure_phase.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements datum and data objects containing closure phases.
  \details To be added
*/

#include "data_closure_phase.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_closure_phase::datum_closure_phase(double u1p, double v1p, double u2p, double v2p, double CPp, double errp, double frequencyp, double tp, std::string Station1p, std::string Station2p, std::string Station3p, std::string Sourcep)
  : CP(CPp), err(errp), 
    u1(u1p), v1(v1p), u2(u2p), v2(v2p), u3(-u1p-u2p), v3(-v1p-v2p), 
    frequency(frequencyp), wavelength(constants::c/frequency), 
    tJ2000(tp),
    Station1(Station1p), Station2(Station2p), Station3(Station3p),
    Source(Sourcep)
{
}

data_closure_phase::data_closure_phase()
{
  _closure_phases.clear();
}

data_closure_phase::data_closure_phase(std::string file_name, const std::string time_type, bool themis_convention)
{
  _closure_phases.clear();
  add_data(file_name, time_type, themis_convention);
}
  
data_closure_phase::data_closure_phase(std::vector<std::string> file_name, const std::vector<std::string> time_type, std::vector<bool> themis_convention)
{
  _closure_phases.clear();
  if (time_type.empty() && themis_convention.empty())
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i]);
  else if (time_type.empty() && !themis_convention.empty())
    for ( size_t i=0; i < file_name.size(); i++ )
      add_data(file_name[i], "HH", themis_convention[i]);  
  else if (!time_type.empty() && themis_convention.empty())
    for ( size_t i=0; i < file_name.size(); i++ )
      add_data(file_name[i], time_type[i], true);
  else
    for (size_t i=0; i<file_name.size(); ++i)
      add_data(file_name[i], time_type[i], themis_convention[i]);
}

void data_closure_phase::add_data(std::string file_name, const std::string time_type, bool themis_convention)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_closure_phase::add_data: Can not open data file" << file_name << " for input. \n";
    std::exit(1);
  }

  if ( themis_convention ){
    std::cout << "Using Themis convention for Fourier Transform phase.\n"; 
  }
  else{
    std::cout << "Using EHT convention for Fourier Transform phase. \n";
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
    
    std::string tmp_src, tmp_tri;
    int tmp_year, tmp_day;
    double tmp_time;
    double tmp_u1, tmp_v1, tmp_u2, tmp_v2;
    double tmp_cp, tmp_err;

    ss_val >> tmp_src >> tmp_year >> tmp_day >> tmp_time >> tmp_tri >> tmp_u1 >> tmp_v1 >> tmp_u2 >> tmp_v2 >> tmp_cp >> tmp_err;

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
      std::cerr << "data_closure_phase: Invalid time format selected\n";
      std::exit(1);
    }

    double tJ2000 = utils::time_J2000(tmp_year,tmp_day,tmp_hour,tmp_min,tmp_sec);

    // In the standard format visibility amplitude data files,
    // u and v are given in Mega-lambda, conver these to units of lambda
    tmp_u1 *= 1e6;
    tmp_v1 *= 1e6;
    tmp_u2 *= 1e6;
    tmp_v2 *= 1e6;

    // Get the station codes for the two participating stations
    std::string tmp_S1, tmp_S2, tmp_S3;
    if (tmp_tri.size()==3) // Using one-letter station codes
    {
      tmp_S1 = tmp_tri[0];
      tmp_S2 = tmp_tri[1];
      tmp_S3 = tmp_tri[2];
    }
    else if (tmp_tri.size()==6) // Using two-letter station codes
    {
      tmp_S1 = tmp_tri[0];
      tmp_S1 += tmp_tri[1];
      tmp_S2 = tmp_tri[2];
      tmp_S2 += tmp_tri[3];
      tmp_S3 = tmp_tri[4];
      tmp_S3 += tmp_tri[5];
    }
    else // Unknown station codes, just copy
    {
      tmp_S1 = tmp_tri;
      tmp_S2 = tmp_tri;
      tmp_S3 = tmp_tri;
    }

    //If data uses eht convention for closure phase multiply by negative one to make consistent with Themis
    if ( !themis_convention )
      tmp_cp *= -1; 

     
    _closure_phases.push_back(new datum_closure_phase(tmp_u1,tmp_v1,tmp_u2,tmp_v2,tmp_cp,tmp_err,230e9,tJ2000,tmp_S1,tmp_S2,tmp_S3,tmp_src));
  }
}

void data_closure_phase::add_data(datum_closure_phase& d)
{
  _closure_phases.push_back(new datum_closure_phase(d));
}

};
