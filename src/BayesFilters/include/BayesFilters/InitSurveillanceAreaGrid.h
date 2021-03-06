#ifndef INITSURVEILLANCEAREAGRID_H
#define INITSURVEILLANCEAREAGRID_H

#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/ParticleSetInitialization.h>

namespace bfl {
    class InitSurveillanceAreaGrid;
}

class bfl::InitSurveillanceAreaGrid : public ParticleSetInitialization
{
public:
    InitSurveillanceAreaGrid(const double surv_x_inf, const double surv_x_sup, const double surv_y_inf, const double surv_y_sup,
                             const unsigned int num_particle_x, const unsigned int num_particle_y) noexcept;

    InitSurveillanceAreaGrid(const double surv_x, const double surv_y,
                             const unsigned int num_particle_x, const unsigned int num_particle_y) noexcept;

    virtual ~InitSurveillanceAreaGrid() noexcept { };

    bool initialize(bfl::ParticleSet& particles) override;

protected:
    double surv_x_inf_;

    double surv_x_sup_;

    double surv_y_inf_;

    double surv_y_sup_;

    double num_particle_x_;

    double num_particle_y_;
};

#endif /* INITSURVEILLANCEAREAGRID_H */
