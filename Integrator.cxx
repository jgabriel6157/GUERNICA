#include "Integrators.hxx"
#include <cmath>

mfem::Vector integrate1D(const std::vector<mfem::GridFunction> &u,
                         const std::vector<double> &vNodes,
                         int power)
{
    int nv = vNodes.size();
    int ndof = u[0].Size();
    double dv = vNodes[1] - vNodes[0];

    mfem::Vector result(ndof);
    result = 0.0;

    for (int i = 0; i < nv; i++)
    {
        const double *f_i = u[i].GetData();
        double v = vNodes[i];
        double w = (i == 0 || i == nv - 1) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);

        double factor = w * std::pow(v, power);
        for (int j = 0; j < ndof; j++)
        {
            result[j] += factor * f_i[j];
        }
    }

    result *= dv / 3.0;
    return result;
}
