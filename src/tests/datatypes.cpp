/*!
  \file datatypes.cpp
  \author 
  \brief main file illustrating read-in of the various EHT observation data
  \details compile using:
  \test Check read-in of several datatypes
 */
//  g++ -O3  -I../util main_test.cpp data_visibility_amplitude.cpp data_closure_phase.cpp data_closure_amplitude.cpp data_flux.cpp ../util/utils.cpp


#include "data_flux.h"
#include "data_polarization_fraction.h"
#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_closure_amplitude.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>

void dump_datum(Themis::datum_flux& d)
{
  std::cerr << std::setw(15) << d.frequency
        	  << std::setw(10) << d.Fnu
        	  << std::setw(10) << d.err
        	  << std::setw(15) << d.tJ2000
        	  << std::setw(5) << d.Source
        	  << '\n';
}

void dump_datum(Themis::datum_visibility_amplitude& d)
{
  std::cerr << std::setw(10) << d.u
      	    << std::setw(10) << d.v
      	    << std::setw(10) << d.V
      	    << std::setw(10) << d.err
      	    << std::setw(15) << d.tJ2000
      	    << std::setw(10) << d.frequency
      	    << std::setw(10) << d.wavelength
      	    << std::setw(5) << d.Station1
      	    << std::setw(5) << d.Station2
      	    << std::setw(5) << d.Source
      	    << '\n';
}


void dump_datum(Themis::datum_closure_phase& d)
{
  std::cerr << std::setw(12) << d.u1
      	    << std::setw(12) << d.v1
      	    << std::setw(12) << d.u2
      	    << std::setw(12) << d.v2
      	    << std::setw(12) << d.u3
      	    << std::setw(12) << d.v3
      	    << std::setw(10) << d.CP
      	    << std::setw(10) << d.err
      	    << std::setw(15) << d.tJ2000
      	    << std::setw(10) << d.frequency
      	    << std::setw(10) << d.wavelength
      	    << std::setw(5) << d.Station1
      	    << std::setw(5) << d.Station2
      	    << std::setw(5) << d.Station3
      	    << std::setw(5) << d.Source
      	    << '\n';
}

void dump_datum(Themis::datum_closure_amplitude& d)
{
  std::cerr << std::setw(12) << d.u1
      	    << std::setw(12) << d.v1
      	    << std::setw(12) << d.u2
      	    << std::setw(12) << d.v2
      	    << std::setw(12) << d.u3
      	    << std::setw(12) << d.v3
      	    << std::setw(12) << d.u4
      	    << std::setw(12) << d.v4
      	    << std::setw(10) << d.CA
      	    << std::setw(10) << d.err
      	    << std::setw(15) << d.tJ2000
      	    << std::setw(10) << d.frequency
      	    << std::setw(10) << d.wavelength
      	    << std::setw(5) << d.Station1
      	    << std::setw(5) << d.Station2
      	    << std::setw(5) << d.Station3
      	    << std::setw(5) << d.Station4
      	    << std::setw(5) << d.Source
      	    << '\n';
}

void dump_datum(Themis::datum_polarization_fraction& d)
{
  std::cerr << std::setw(12) << d.u
      	    << std::setw(12) << d.v
      	    << std::setw(10) << d.mbreve_amp
      	    << std::setw(10) << d.err
      	    << std::setw(15) << d.tJ2000
      	    << std::setw(10) << d.frequency
      	    << std::setw(10) << d.wavelength
      	    << std::setw(5) << d.Station1
      	    << std::setw(5) << d.Station2
      	    << std::setw(5) << d.Source
      	    << '\n';
}

int main(int argc,char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  Themis::datum_flux f1(230e9,1,0.1);
  dump_datum(f1);

  Themis::datum_flux f2(230e9,1,0.1,2e9,"SGRA");
  dump_datum(f2);

  Themis::data_flux f("../../eht_data/spec_data.d");
  
  std::cerr << '\n'
	    << "Size of f " << f.size() << '\n';
  for (size_t i=0; i<f.size(); ++i)
    dump_datum(f.datum(i));
  std::cerr << "\n\n";

  Themis::datum_visibility_amplitude d1(10,20,1,0.1);
  dump_datum(d1);

  Themis::datum_visibility_amplitude d2(10,20,1,0.1,345e9,3.24592e8,"JCMT","ALMA-A");
  dump_datum(d2);

  Themis::datum_visibility_amplitude d3=d2;
  dump_datum(d3);

  Themis::data_visibility_amplitude d07("../../eht_data/VM_2007_100.d");
  d07.add_data("../../eht_data/VM_2007_101.d");

  
  std::vector<std::string> vm_files;
  vm_files.push_back("../../eht_data/VM_2009_095.d");
  vm_files.push_back("../../eht_data/VM_2009_096.d");
  vm_files.push_back("../../eht_data/VM_2009_097.d");
  Themis::data_visibility_amplitude d09(vm_files);

  std::cerr << '\n'
	    << "Size of d07 " << d07.size() << '\n';
  for (size_t i=0; i<d07.size(); ++i)
    dump_datum(d07.datum(i));

  std::cerr << '\n'
	    << "Size of d09 " << d09.size() << '\n';
  for (size_t i=0; i<d09.size(); ++i)
    dump_datum(d09.datum(i));


  std::cerr << "\n\n";
  Themis::datum_closure_phase c1(10,20,1,-3.14,27.0,0.5);
  dump_datum(c1);

  Themis::data_closure_phase c09("../../eht_data/CP_2009_093.d");
  std::cerr << '\n'
	    << "Size of c09 " << c09.size() << '\n';
  for (size_t i=0; i<c09.size(); ++i)
    dump_datum(c09.datum(i));
  


  std::cerr << "\n\n";
  Themis::datum_closure_amplitude a1(10,20,1,2,15,24,1.142,0.242);
  dump_datum(a1);


  // FRACTIONAL POLARIZATION (MBREVE)
  
  std::cerr << "\n\n";
  std::vector<std::string> mbreve_files;

  mbreve_files.push_back("../../eht_data/MBREVE_2013_080.d");
  mbreve_files.push_back("../../eht_data/MBREVE_2013_081.d");
  mbreve_files.push_back("../../eht_data/MBREVE_2013_082.d");
  mbreve_files.push_back("../../eht_data/MBREVE_2013_085.d");
  mbreve_files.push_back("../../eht_data/MBREVE_2013_086.d");
  
  Themis::data_polarization_fraction mbreve2013(mbreve_files);

  std::cerr << '\n' << "Size of mbreve2013 " << mbreve2013.size() << '\n';
  
  for (size_t i=0; i<mbreve2013.size(); ++i)
    dump_datum(mbreve2013.datum(i));
    


  MPI_Finalize();

  return 0;
}
