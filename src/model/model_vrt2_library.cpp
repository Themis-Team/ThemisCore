/*!
  \file model_vrt2_library.cpp
  \author Avery Broderick
  \date  July, 2017
  \brief Implements the asymmetric model_vrt2_library image class.
  \details To be added
*/

#include "model_vrt2_library.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>


namespace Themis {

  model_vrt2_library::model_vrt2_library(std::string index_file_name, std::string pmap_file_format, size_t Nindex, double Mcm, double Dcm)
    : _model(Mcm,Dcm), _pmap_file_format(pmap_file_format), _Nparams(Nindex)
  {
    // Read index file (job_list file)
    std::ifstream in(index_file_name);
    if (!in.is_open())
    {
      std::cerr << "model_vrt2_library::model_vrt2_library Could not open " << index_file_name << '\n';
      std::exit(1);
    }
    _index_list.resize(_Nparams);
    _parameter_list.resize(_Nparams);
    int itmp;
    double dtmp;
    in.ignore(4096,'\n'); // Read header off
    for (in>>itmp; !in.eof(); in>>itmp)
    {
      _index_list[0].push_back(itmp);
      for (size_t j=1; j<_Nparams; ++j)
      {
	in >> itmp;
	_index_list[j].push_back(itmp);
      }

      for (size_t j=0; j<_Nparams; ++j)
      {
	in >> dtmp;
	_parameter_list[j].push_back(dtmp);
      }

      in.ignore(4096,'\n');
    }    
  }

  std::vector<int> model_vrt2_library::get_library_indexes(std::vector<double> params)
  {
    double L2diff=0.0;
    double L2diffmin=0.0;
    size_t idiffmin=0;
    for (size_t i=0; i<_index_list[0].size(); ++i)
    {
      L2diff = 0.0;
      for (size_t j=0; j<_Nparams; ++j)
	L2diff += std::pow(params[j]-_parameter_list[j][i],2);

      if (i==0 || L2diff<L2diffmin)
      {
	L2diffmin=L2diff;
	idiffmin=i;
      }
    }

    std::vector<int> indexes;
    for (size_t j=0; j<_Nparams; ++j)
      indexes.push_back(_index_list[j][idiffmin]);
    
    return indexes;
  }

  void model_vrt2_library::generate_model(std::vector<double> parameters)
  {
    std::vector<int> index=get_library_indexes(parameters);

    // Construct file name.  A potential weak spot, but unlikely to matter in practice
    // since libraries don't currently exist with more than 3 parameters.
    char buff[4096];
    if (_Nparams==0)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0]);
    else if (_Nparams==1)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1]);
    else if (_Nparams==2)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1],index[2]);
    else if (_Nparams==3)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1],index[2],index[3]);
    else if (_Nparams==4)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1],index[2],index[3],index[4]);
    else if (_Nparams==5)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1],index[2],index[3],index[4],index[5]);
    else if (_Nparams==6)
      std::sprintf(buff,_pmap_file_format.c_str(),index[0],index[1],index[2],index[3],index[4],index[5],index[6]);
    std::string pmap_file_name(buff);
    
    _model.set_pmap_file(pmap_file_name);

    for (size_t j=0; j<_Nparams; ++j)
      parameters[j] = index[j];
    
    _model.generate_model(parameters);
  }

  double model_vrt2_library::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    return ( _model.visibility_amplitude(d,acc) );
  }

  double model_vrt2_library::closure_phase(datum_closure_phase& d, double acc)
  {
    return ( _model.closure_phase(d,acc) );
  }

  double model_vrt2_library::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    return ( _model.closure_amplitude(d,acc) );
  }

  void model_vrt2_library::get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
  {
    _model.get_image(alpha,beta,I);
  }

  
};
