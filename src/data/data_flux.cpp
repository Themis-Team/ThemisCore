/*!
  \file data_flux.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements datum and data objects containing fluxes.
  \details To be added
*/

#include "data_flux.h"

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>

#include "constants.h"
#include "utils.h"

namespace Themis {

datum_flux::datum_flux(double frequencyp, double Fnup, double errp, double tp, std::string Sourcep)
  : Fnu(Fnup), err(errp), frequency(frequencyp), 
    wavelength(constants::c/frequency), 
    tJ2000(tp),
    Source(Sourcep)
{
}

data_flux::data_flux()
{
  _fluxes.clear();
}

data_flux::data_flux(std::string file_name)
{
  _fluxes.clear();
  add_data(file_name);
}
  
data_flux::data_flux(std::vector<std::string> file_name)
{
  _fluxes.clear();
  for (size_t i=0; i<file_name.size(); ++i)
    add_data(file_name[i]);
}

void data_flux::add_data(std::string file_name)
{
  std::ifstream ifile;
  ifile.open(file_name.c_str());
  if(!ifile.is_open())
  {
    std::cerr << "data_flux::add_data: Can not open data file" << file_name << " for input. \n";
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
    
    double tmp_lognu, tmp_Fnu, tmp_err;

    ss_val >> tmp_lognu >> tmp_Fnu >> tmp_err;
    
    double tmp_nu = std::pow(10.0,tmp_lognu);
    
    _fluxes.push_back(new datum_flux(tmp_nu,tmp_Fnu,tmp_err));
  }
}

void data_flux::add_data(datum_flux& d)
{
  _fluxes.push_back(new datum_flux(d));
}
};
