#include "magnetic_field.h"

namespace VRT2 {
MagneticField::MagneticField(Metric& g)
  : _g(g), _b(_g)
{
}
};
