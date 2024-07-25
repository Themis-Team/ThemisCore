/*!
  \file image_family_error.cpp
  \author Avery E. Broderick
  \date  December, 2018
  \brief Source file for a utility class that generates an approximation of the systemtic error as measured by the variance among images.
*/

#include "image_family_error.h"

namespace Themis {

  image_family_static_error::image_family_static_error()
    : _approximation_type(1)
  {
    // Initialize the data objects (zero size)
    _data_v = new Themis::data_visibility;
    _data_va = new Themis::data_visibility_amplitude;
    _data_cp = new Themis::data_closure_phase;
    _data_ca = new Themis::data_closure_amplitude;
  }

  image_family_static_error::image_family_static_error(std::vector<std::string> image_file_name_list, std::string README_file_name, std::vector<double> p, size_t Nr, size_t Nphi, double umax)
    : _approximation_type(1)
  {
    generate_error_estimates(image_file_name_list,README_file_name,p,Nr,Nphi,umax);

    // Initialize the data objects (zero size)
    _data_v = new Themis::data_visibility;
    _data_va = new Themis::data_visibility_amplitude;
    _data_cp = new Themis::data_closure_phase;
    _data_ca = new Themis::data_closure_amplitude;
  }

  void image_family_static_error::generate_error_estimates(std::vector<std::string> image_file_name_list, std::string README_file_name, std::vector<double> p, size_t Nr, size_t Nphi, double umax)
  {
    //std::cerr << "Started in ifse\n";

    // Generate means and variances of the visibility amplitudes from the images
    // 1. Make space and initialize to zero
    double *vis_amp_mean = new double[Nr*Nphi];
    double *vis_amp_var = new double[Nr*Nphi];

    double *vis_amp_mean_single = new double[Nr*Nphi];
    double *vis_amp_var_single = new double[Nr*Nphi];

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for (size_t k=0; k<Nr*Nphi; ++k)
    {
      vis_amp_mean[k] = vis_amp_var[k] = 0.0;
    }

    // 2. Average at a number of representative points in u-v space
    for (size_t j=0; j<image_file_name_list.size(); ++j)
    {
      if (world_rank==int(j)%world_size)
      {
	std::cerr << "Rank " << world_rank << "> image_family_static_error: Reading image number " << j << std::endl;
	model_image_score img(image_file_name_list[j],README_file_name);
	img.generate_model(p);
	double norm=0.0;
	for (size_t k=0, mr=0; mr<Nr; ++mr)
	  for (size_t mp=0; mp<Nphi; ++mp, ++k)
	  {
	    double umag = umax*1e9*double(mr+1)/double(Nr);
	    double phi = M_PI*double(mp)/double(Nphi);
	    datum_visibility_amplitude d(umag*std::cos(phi),umag*std::sin(phi),1,0);
	    double val = img.visibility_amplitude(d,0);
	    
	    vis_amp_mean_single[k] = val;
	    vis_amp_var_single[k] = val*val;

	    norm += val;
	  }
	norm = 1.0/norm;
	for (size_t k=0, mr=0; mr<Nr; ++mr)
	  for (size_t mp=0; mp<Nphi; ++mp, ++k)
	  {
	    vis_amp_mean[k] += vis_amp_mean_single[k] * norm;
	    vis_amp_var[k] += vis_amp_var_single[k] * norm*norm;
	  }
      }
    }
    double *buff1 = new double[Nr*Nphi];
    double *buff2 = new double[Nr*Nphi];
    MPI_Allreduce(&vis_amp_mean[0],&buff1[0],int(Nr*Nphi),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&vis_amp_var[0],&buff2[0],int(Nr*Nphi),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    // 3. Construct mean and variance appropriately at each test point
    for (size_t k=0; k<Nr*Nphi; ++k)
    {
      vis_amp_mean[k] = buff1[k]/double(image_file_name_list.size());
      vis_amp_var[k] = (buff2[k] - vis_amp_mean[k]*vis_amp_mean[k]*image_file_name_list.size())/(double(image_file_name_list.size())-1.);
    }

    //  Renorm the mean back to p[0]
    double norm = 0.0;
    for (size_t mp=0; mp<Nphi; ++mp)
      norm += vis_amp_mean[mp];
    norm = p[0]*Nphi/norm;
    for (size_t k=0; k<Nr*Nphi; ++k)
    {
      vis_amp_mean[k] *= norm;
      vis_amp_var[k] *= norm*norm;
    }

    // 4. Average down to get the SNR and error as a function of radius
    std::vector<double> umag(Nr), va_var(Nr), va_frac_var(Nr);
    for (size_t k=0, mr=0; mr<Nr; ++mr)
    {
      umag[mr] = umax*1e9*double(mr+1)/double(Nr);
      va_var[mr] = 0.0;
      va_frac_var[mr] = 0.0;
      for (size_t mp=0; mp<Nphi; ++mp, ++k)
      {
	va_var[mr] += vis_amp_var[k];
	va_frac_var[mr] += vis_amp_var[k]/(vis_amp_mean[k]*vis_amp_mean[k]);
      }
      
      va_var[mr] = va_var[mr]/Nphi;
      va_frac_var[mr] = va_frac_var[mr]/Nphi;
    }

    // 5. Set the 1D interpolator tables
    _vis_amp_var.set_tables(umag,va_var);
    _vis_amp_frac_var.set_tables(umag,va_frac_var);

    // 6. Average down to get a uniform measure of the variance and mean (superfluous)
    _const_va_var = _const_va_frac_var = 0.0;
    for (size_t mr=0; mr<Nr; ++mr)
    {
      _const_va_var += va_var[mr];
      _const_va_frac_var += va_frac_var[mr];
    }
    _const_va_var /= Nr;
    _const_va_frac_var /= Nr;


    // DEBUG output
    if (world_rank==0)
    {
      std::ofstream vout("VA_full.d");
      for (size_t k=0, mr=0; mr<Nr; ++mr)
      {
	for (size_t mp=0; mp<Nphi; ++mp, ++k)
        {
	  double umag = umax*double(mr+1)/double(Nr);
	  double phi = M_PI*double(mp)/double(Nphi);
	  
	  vout << std::setw(15) << umag
	       << std::setw(15) << phi
	       << std::setw(15) << vis_amp_mean[k]
	       << std::setw(15) << std::sqrt(vis_amp_var[k])
	       << std::setw(15) << _const_va_var
	       << std::setw(15) << _const_va_frac_var
	       << std::setw(15) << _vis_amp_var(umag*1e9)
	       << std::setw(15) << _vis_amp_frac_var(umag*1e9)
	       << std::setw(15) << va_var[mr]
	       << std::setw(15) << va_frac_var[mr]
	       << std::endl;
	}
	vout << std::endl;
      }
    }


    // Clean up
    delete[] vis_amp_mean;
    delete[] vis_amp_var;
    delete[] vis_amp_mean_single;
    delete[] vis_amp_var_single;
    delete[] buff1;
    delete[] buff2;
    //std::cerr << "Finished creation of ifse\n";

  }

  image_family_static_error::~image_family_static_error()
  {
    delete _data_v;
    delete _data_va;
    delete _data_cp;
    delete _data_ca;
  }
  
  Themis::data_visibility& image_family_static_error::data_visibility(Themis::data_visibility& d) //, std::vector<double> p)
  {
    return d;
  }
  
  Themis::data_visibility_amplitude& image_family_static_error::data_visibility_amplitude(Themis::data_visibility_amplitude& d) //, std::vector<double> p)
  {
    // Delete and remake data object
    delete _data_va;
    _data_va = new Themis::data_visibility_amplitude;
    
    // Generate new local data object
    for (size_t k=0; k<d.size(); ++k)
    {
      double umag = std::sqrt( d.datum(k).u* d.datum(k).u + d.datum(k).v*d.datum(k).v );
      double theory_var = 0.0;
      if (_approximation_type==0)
	theory_var = _const_va_var;
      else if (_approximation_type==1)
	theory_var = _vis_amp_var(umag);
      double new_error = std::sqrt( d.datum(k).err*d.datum(k).err + theory_var );
      datum_visibility_amplitude dnew(d.datum(k).u,d.datum(k).v,d.datum(k).V,new_error,d.datum(k).frequency,d.datum(k).tJ2000,d.datum(k).Station1,d.datum(k).Station2,d.datum(k).Source);
      _data_va->add_data(dnew);
    }

    return (*_data_va);
  }

  Themis::data_closure_phase& image_family_static_error::data_closure_phase(Themis::data_closure_phase& d) //, std::vector<double> p)
  {
    // Delete and remake data object
    delete _data_cp;
    _data_cp = new Themis::data_closure_phase;

    // Generate new local data object
    for (size_t k=0; k<d.size(); ++k)
    {
      double r2d2 = (180.*180.)/(M_PI*M_PI);
      double theory_var = 0.0;
      if (_approximation_type==0)
	theory_var = 3.0*_const_va_frac_var * r2d2;
      else if (_approximation_type==1)
	theory_var = ( _vis_amp_frac_var( std::sqrt( d.datum(k).u1* d.datum(k).u1 + d.datum(k).v1*d.datum(k).v1 ) )
		       + _vis_amp_frac_var( std::sqrt( d.datum(k).u2* d.datum(k).u2 + d.datum(k).v2*d.datum(k).v2 ) )
		       + _vis_amp_frac_var( std::sqrt( d.datum(k).u3* d.datum(k).u3 + d.datum(k).v3*d.datum(k).v3 ) ) )* r2d2;
      
      double new_error = std::sqrt( d.datum(k).err*d.datum(k).err + theory_var );
      datum_closure_phase dnew(d.datum(k).u1,d.datum(k).v1,d.datum(k).u2,d.datum(k).v2,d.datum(k).CP,new_error,d.datum(k).frequency,d.datum(k).tJ2000,d.datum(k).Station1,d.datum(k).Station2,d.datum(k).Station3,d.datum(k).Source);
      _data_cp->add_data(dnew);
    }

    return (*_data_cp);
  }
  
  Themis::data_closure_amplitude& image_family_static_error::data_closure_amplitude(Themis::data_closure_amplitude& d) //, std::vector<double> p)
  {
    return d;
  }
    
  


};
