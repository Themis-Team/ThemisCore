/*!
  \file utils.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements a variety of utility functions within Themis.
  \details To be added
*/

#include "utils.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cstring>

namespace Themis {

  bool utils::isfile(const std::string& name)
  {
    // based on https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-14-17-c
      struct stat buffer;   
      return (stat (name.c_str(), &buffer) == 0); 
  }

  /// \bug Daylight saving time not properly handled
  double utils::time_J2000(int year, int day, int hour, int min, double sec)
  {
    // Generate a struct tm object with the right time
    struct tm timeinfo;
    
    timeinfo.tm_year = year-1900;
    timeinfo.tm_mon = 0;
    timeinfo.tm_mday = day;
    timeinfo.tm_hour = hour;
    timeinfo.tm_min = min;
    timeinfo.tm_sec = 0;
    timeinfo.tm_isdst = 0;
    
    // Generate a time_t object from this
    time_t timeobs = mktime(&timeinfo);
    // Keep track of when daylight savings time is on
    int dstshift = timeinfo.tm_isdst;
    
    // Get the time since 2000  J2000 epoch is defined by 12 UTC on Jan 1 2000
    timeinfo.tm_year = 2000-1900;
    timeinfo.tm_mon = 0;
    timeinfo.tm_mday = 1;
    timeinfo.tm_hour = 12;
    timeinfo.tm_min = 0;
    timeinfo.tm_sec = 0;
    timeinfo.tm_isdst = 0;
    
    time_t time2000 = mktime(&timeinfo);
    dstshift -= timeinfo.tm_isdst;
    
    // std::cout << "[utils.cpp] utils::time_J2000() : dstshift=" <<dstshift<<", timeinfo.tm_year,timeinfo.tm_mon,timeinfo.tm_mday,timeinfo.tm_hour:"<<timeinfo.tm_year<<" "<<timeinfo.tm_mon<<" "<<timeinfo.tm_mday<<" "<<timeinfo.tm_hour<< std::endl;
    // std::cout << "[utils.cpp] utils::time_J2000() : dstshift=" << dstshift << std::endl;
    
    // Return difference of times minus a shift to fix up DST differences.
    //return ( difftime(timeobs,time2000)-3600*dstshift+sec );
    return ( difftime(timeobs,time2000)+sec );
  }

  /// \warning File names limited to 4096 characters.
  std::string utils::get_file_extension(std::string file_name)
  {
    char test[4096];
    std::strcpy(test,file_name.c_str());
    char *word;
    std::vector<std::string> tokens;
    for (word = std::strtok(test, "."); word; word = std::strtok(NULL, "."))
      tokens.push_back(word);
    return tokens.back();
  }


  std::string utils::global_path(std::string file_name)
  {
    // Set default
    std::string themis_path=THEMISPATH;

    // Check to see if THEMISPATH is set
    char* envpath = std::getenv("THEMISPATH");
    if (envpath!=NULL) {
      themis_path=std::string(envpath);
      std::cout << "Epath= " << envpath << std::endl;
    }

    // Prepend the filename
    return (themis_path+"/"+file_name);
  }


  std::vector<std::string> utils::station_codes(std::string listname)
  {
    std::vector<std::string> station_codes;

    if (listname=="HOPS 2017")
    {
      station_codes.push_back("A"); // ALMA phased
      station_codes.push_back("B"); // ALMA single
      station_codes.push_back("X"); // APEX
      station_codes.push_back("G"); // GLT
      station_codes.push_back("J"); // JCMT
      station_codes.push_back("K"); // Kitt Peak
      station_codes.push_back("L"); // LMT
      station_codes.push_back("N"); // NOEMA phased
      station_codes.push_back("M"); // NOEMA single
      station_codes.push_back("P"); // Pico Veleta
      station_codes.push_back("C"); // SMA single
      station_codes.push_back("S"); // SMA phased
      station_codes.push_back("R"); // SMA reference
      station_codes.push_back("Z"); // SMT
      station_codes.push_back("Y"); // SPT
    }
    else if (listname=="Two-letter 2017")
    {
      station_codes.push_back("Aa"); // ALMA phased
      station_codes.push_back("Aq"); // ALMA single
      station_codes.push_back("Ax"); // APEX
      station_codes.push_back("Gl"); // GLT
      station_codes.push_back("Mm"); // JCMT
      station_codes.push_back("Kt"); // Kitt Peak
      station_codes.push_back("Lm"); // LMT
      station_codes.push_back("Na"); // NOEMA phased
      station_codes.push_back("Nq"); // NOEMA single
      station_codes.push_back("Pv"); // Pico Veleta
      station_codes.push_back("Sq"); // SMA single
      station_codes.push_back("Sw"); // SMA phased
      station_codes.push_back("Sr"); // SMA reference
      station_codes.push_back("Mg"); // SMT
      station_codes.push_back("Sz"); // SPT
    }
    else if (listname=="uvfits 2017")
    {
      station_codes.push_back("AA"); // ALMA phased
      station_codes.push_back("AP"); // APEX
      station_codes.push_back("AZ"); // SMT
      station_codes.push_back("JC"); // JCMT
      station_codes.push_back("LM"); // LMT
      station_codes.push_back("PV"); // Pico Veleta
      station_codes.push_back("SM"); // SMA phased
      station_codes.push_back("SP"); // SPT
    }
    else if (listname=="uvfits 2018") //
    {
      station_codes.push_back("AA"); // ALMA phased
      station_codes.push_back("AX"); // APEX
      station_codes.push_back("GL"); // GLT
      station_codes.push_back("LM"); // LMT
      station_codes.push_back("MG"); // SMT
      station_codes.push_back("MM"); // JCMT
      station_codes.push_back("PV"); // Pico Veleta
      station_codes.push_back("SW"); // SMA phased
      station_codes.push_back("SZ"); // SPT
    }
    else
    {
      std::cerr << "Station code list " << listname << " not recognized.\n";
      std::exit(1);
    }

    return station_codes;
  }
  
};



