/*!
  \file read_data.h
  \author Paul Tiede Avery Broderick
  \date Oct 21, 2021
  \brief a number of utilities made so that we can read in data consistently across models
*/ 


#ifndef Themis_READ_DATA_H
#define Themis_READ_DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <cstring>


namespace Themis {
  namespace utils {

    /*!
      \brief Reads in the vfile usually passed in a themis driver in such a way to prevent MPI clobbering
      \param v_file_name_list The buffer that will hold the results
      \param v_file The file name to read in
      \param comm The MPI communicator you will be reading with (This is usually comm)
    */
    void read_vfile_mpi(std::vector<std::string>& v_file_name_list, std::string v_file, MPI_Comm comm)
    {
      int rank, size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);

  
      std::string v_file_name;
      int strlng;
      if (rank==0)
      {
	std::ifstream vin(v_file);

	for (vin>>v_file_name; !vin.eof(); vin>>v_file_name) 
	  {
	    v_file_name_list.push_back(v_file_name);
	    std::cout<<"Reading in data file: "<<v_file_name<<std::endl;
	  }
	// strlng = v_file_name.length()+1; // Unnecessary
      }
      int ibuff = v_file_name_list.size();

      if (rank==0) std::cout<<"Read in "<<ibuff<<" data files"<<std::endl;

      MPI_Bcast(&ibuff,1,MPI_INT,0, comm);
      for (int i=0; i<ibuff; ++i)
      {
	if (rank==0)
	  strlng = v_file_name_list[i].length()+1;
	MPI_Bcast(&strlng,1,MPI_INT,0, comm);
	
	char* cbuff = new char[strlng];
	if (rank==0)
	  strcpy(cbuff,v_file_name_list[i].c_str());
	MPI_Bcast(&cbuff[0],strlng,MPI_CHAR,0, comm);
	if (rank>0)
	  v_file_name_list.push_back(std::string(cbuff));
	delete[] cbuff;
      }
    }

  };
};
  


#endif
