/*! 
  \file examples/reading_data.cpp
  \author Avery E. Broderick
  \date June, 2017
  \brief Example of how to read and access common EHT data types.
  
  \details In Themis, data is encapsulated in data objects specified for 
   data type.  These include visibility amplitudes, closure phases, closure
   amplitudes, and fluxes.  These data types contain the data values along
   with particular details about the data collected (e.g., frequency,
   time, station, baseline, etc.).  Reading in the data may be accomplished
   at construction, afterward via the "add_data" command, or by adding
   data points directly.  Here we provide a handful of examples of each
   type.
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_flux.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
  // Read in visibility amplitude data from day 100 of 2007 contained
  // in the file located in Themis/eht_data/VM_2007_100.d
  Themis::data_visibility_amplitude VM_2007("../../eht_data/VM_2007_100.d");

  // Add the data from day 101 of 2007 contained in the file located
  // in Themis/eht_data/VM_2007_101.d
  VM_2007.add_data("../../eht_data/VM_2007_101.d");

  // Add a specific data point, here a single-dish flux of 2.5+-0.25 Jy
  Themis::datum_visibility_amplitude VM(0,0,2.5,0.25);
  VM_2007.add_data(VM);
  
  // Read in closure phase data from day 96 of 2009 contained in the
  // file located in Themis/eht_data/CP_2009_096.d
  Themis::data_closure_phase CP_2009_096("../../eht_data/CP_2009_096.d", "HH", true);

  // Read in flux data contained in the file located in
  // Themis/eht_data/spec_data.d
  Themis::data_flux SED("../../eht_data/spec_data.d");

  // Now print all of the data points out just to see what we have read in.
  // Note that this is not generally necessary, but here is instructive.
  // For the visibility data, data points 0-6 are from day 100, 7-18 are
  // from day 101, and 19 is the additional single dish data point.
  std::cout << "\nVisibility amplitude data:\n";
  std::cout << std::setw(5) << "#"
	          << std::setw(12) << "u (lambda)"
	          << std::setw(12) << "v (lambda)"
            << std::setw(12) << "|V| (Jy)"
    	      << std::setw(12) << "error (Jy)"
	          << std::endl;

  for (size_t j=0; j<VM_2007.size(); ++j)
    std::cout << std::setw(5) << j
      	      << std::setw(12) << VM_2007.datum(j).u
      	      << std::setw(12) << VM_2007.datum(j).v
      	      << std::setw(12) << VM_2007.datum(j).V
      	      << std::setw(12) << VM_2007.datum(j).err
      	      << std::endl;
  std::cout << "\n";

  std::cout << "\nClosure phase data:\n";
  std::cout << std::setw(5) << "#"
      	    << std::setw(12) << "u1 (lambda)"
      	    << std::setw(12) << "v1 (lambda)"
      	    << std::setw(12) << "u2 (lambda)"
      	    << std::setw(12) << "v2 (lambda)"
      	    << std::setw(12) << "CP (deg)"
    	      << std::setw(12) << "error (deg)"
	          << std::endl;

  for (size_t j=0; j<CP_2009_096.size(); ++j)
    std::cout << std::setw(5) << j
      	      << std::setw(12) << CP_2009_096.datum(j).u1
      	      << std::setw(12) << CP_2009_096.datum(j).v1
      	      << std::setw(12) << CP_2009_096.datum(j).u2
      	      << std::setw(12) << CP_2009_096.datum(j).v2
      	      << std::setw(12) << CP_2009_096.datum(j).CP
      	      << std::setw(12) << CP_2009_096.datum(j).err
      	      << std::endl;
  std::cout << "\n";

  std::cout << "\nFlux data:\n";
  std::cout << std::setw(5) << "#"
      	    << std::setw(12) << "nu (Hz)"
	          << std::setw(12) << "Fnu (Jy)"
    	      << std::setw(12) << "error (Jy)"
	          << std::endl;

  for (size_t j=0; j<SED.size(); ++j)
    std::cout << std::setw(5) << j
      	      << std::setw(12) << SED.datum(j).frequency
	            << std::setw(12) << SED.datum(j).Fnu
	            << std::setw(12) << SED.datum(j).err
	            << std::endl;
  std::cout << "\n\n";
  
  return 0;
}

/*! 
  \file examples/reading_data.cpp
  \details 
  
  \code
  #include "data_visibility_amplitude.h"
  #include "data_closure_phase.h"
  #include "data_flux.h"
  #include <iostream>
  #include <iomanip>

  int main(int argc, char* argv[])
  {
    // Read in visibility amplitude data from day 100 of 2007 contained
    // in the file located in Themis/eht_data/VM_2007_100.d
    Themis::data_visibility_amplitude VM_2007("../../eht_data/VM_2007_100.d");

    // Add the data from day 101 of 2007 contained in the file located
    // in Themis/eht_data/VM_2007_101.d
    VM_2007.add_data("../../eht_data/VM_2007_101.d");

    // Add a specific data point, here a single-dish flux of 2.5+-0.25 Jy
    Themis::datum_visibility_amplitude VM(0,0,2.5,0.25);
    VM_2007.add_data(VM);
    
    // Read in closure phase data from day 96 of 2009 contained in the
    // file located in Themis/eht_data/CP_2009_096.d
    Themis::data_closure_phase CP_2009_096("../../eht_data/CP_2009_096.d");

    // Read in flux data contained in the file located in
    // Themis/eht_data/spec_data.d
    Themis::data_flux SED("../../eht_data/spec_data.d");

    // Now print all of the data points out just to see what we have read in.
    // Note that this is not generally necessary, but here is instructive.
    // For the visibility data, data points 0-6 are from day 100, 7-18 are
    // from day 101, and 19 is the additional single dish data point.
    std::cout << "\nVisibility amplitude data:\n";
    std::cout << std::setw(5) << "#"
  	          << std::setw(12) << "u (lambda)"
  	          << std::setw(12) << "v (lambda)"
              << std::setw(12) << "|V| (Jy)"
      	      << std::setw(12) << "error (Jy)"
  	          << std::endl;

    for (size_t j=0; j<VM_2007.size(); ++j)
      std::cout << std::setw(5) << j
        	      << std::setw(12) << VM_2007.datum(j).u
        	      << std::setw(12) << VM_2007.datum(j).v
        	      << std::setw(12) << VM_2007.datum(j).V
        	      << std::setw(12) << VM_2007.datum(j).err
        	      << std::endl;
    std::cout << "\n";

    std::cout << "\nClosure phase data:\n";
    std::cout << std::setw(5) << "#"
        	    << std::setw(12) << "u1 (lambda)"
        	    << std::setw(12) << "v1 (lambda)"
        	    << std::setw(12) << "u2 (lambda)"
        	    << std::setw(12) << "v2 (lambda)"
        	    << std::setw(12) << "CP (deg)"
      	      << std::setw(12) << "error (deg)"
  	          << std::endl;

    for (size_t j=0; j<CP_2009_096.size(); ++j)
      std::cout << std::setw(5) << j
        	      << std::setw(12) << CP_2009_096.datum(j).u1
        	      << std::setw(12) << CP_2009_096.datum(j).v1
        	      << std::setw(12) << CP_2009_096.datum(j).u2
        	      << std::setw(12) << CP_2009_096.datum(j).v2
        	      << std::setw(12) << CP_2009_096.datum(j).CP
        	      << std::setw(12) << CP_2009_096.datum(j).err
        	      << std::endl;
    std::cout << "\n";

    std::cout << "\nFlux data:\n";
    std::cout << std::setw(5) << "#"
        	    << std::setw(12) << "nu (Hz)"
  	          << std::setw(12) << "Fnu (Jy)"
      	      << std::setw(12) << "error (Jy)"
  	          << std::endl;

    for (size_t j=0; j<SED.size(); ++j)
      std::cout << std::setw(5) << j
        	      << std::setw(12) << SED.datum(j).frequency
  	            << std::setw(12) << SED.datum(j).Fnu
  	            << std::setw(12) << SED.datum(j).err
  	            << std::endl;
    std::cout << "\n\n";
    
    return 0;
  }
  \endcode
*/

