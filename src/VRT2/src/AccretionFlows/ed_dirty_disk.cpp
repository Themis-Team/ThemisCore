#include "ed_dirty_disk.h"

namespace VRT2 {
ED_DirtyDisk::ED_DirtyDisk(ElectronDensity& ed, Metric& g, double rISCO, double amp, std::string fname, int Nlr, int Nth, int Nph)
: _ed_raw(ed), _amp(amp), _rISCO(rISCO), _Nlr(Nlr), _Nth(Nth), _Nph(Nph)
//: _ed_raw(ed), _g(g), _amp(amp), _rISCO(rISCO), _Nlr(Nlr), _Nth(Nth), _Nph(Nph)
{
  _dd = new double**[_Nlr];
  for (int ilr=0; ilr<_Nlr; ++ilr)
  {
    _dd[ilr] = new double*[_Nth];
    for (int ith=0; ith<_Nth; ++ith)
      _dd[ilr][ith] = new double[_Nph];
  }

  std::ifstream in(fname.c_str());
  if (! in.is_open())
  {
    std::cerr << "Could not open " << fname << " in ED_DirtyDisk\n";
    exit(0);
  }

  double rmin, rmax;
  in >> rmin;
  in >> rmax;
  _lrmin = std::log(rmin);
  _lrmax = std::log(rmax);
  in.ignore(4096,'\n');

  double junk;
  double min=0.0, max=0.0;
  for (int ilr=0; ilr<_Nlr; ++ilr)
    for (int ith=0; ith<_Nth; ++ith)
      for (int iph=0; iph<_Nph; ++iph)
      {
	in >> junk;
	in >> junk;
	in >> junk;
	in >> _dd[ilr][ith][iph];
	in.ignore(4096,'\n');

	if (_dd[ilr][ith][iph]<min)
	  min = _dd[ilr][ith][iph];
	if (_dd[ilr][ith][iph]>max)
	  max = _dd[ilr][ith][iph];
      }

  in.close();

  // Renormalize _dd
  double norm = 2.0/(max-min);
  for (int ilr=0; ilr<_Nlr; ++ilr)
    for (int ith=0; ith<_Nth; ++ith)
      for (int iph=0; iph<_Nph; ++iph)
	_dd[ilr][ith][iph] = norm*(_dd[ilr][ith][iph]-min) - 1.0;

  /*
  for (int ith=0; ith<_Nth; ++ith)
  {
    for (int ilr=0; ilr<_Nlr; ++ilr)
    {
      for (int iph=0; iph<_Nph; ++iph)
	std::cout << std::setw(15) << ith
		  << std::setw(15) << ilr
		  << std::setw(15) << iph
		  << std::setw(15) << _dd[ilr][ith][iph]
		  << '\n';
      std::cout << '\n';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  */



  _dlr = (_lrmax-_lrmin)/(_Nlr-1);
  _dth = M_PI/_Nth;
  _dph = 2.0*M_PI/_Nth;
}
ED_DirtyDisk::~ED_DirtyDisk()
{
  for (int ilr=0; ilr<_Nlr; ++ilr)
  {
    for (int ith=0; ith<_Nth; ++ith)
      delete[] _dd[ilr][ith];
    delete[] _dd[ilr];
  }
  delete[] _dd;
}
};
