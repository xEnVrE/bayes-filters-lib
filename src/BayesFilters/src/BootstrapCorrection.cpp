#include <BayesFilters/BootstrapCorrection.h>

#include <cmath>
#include <utility>

using namespace bfl;
using namespace Eigen;


BoostrapCorrection::BoostrapCorrection() noexcept { }


BoostrapCorrection::~BoostrapCorrection() noexcept { }


void BoostrapCorrection::correctStep(const ParticleSet& pred_particles, ParticleSet& cor_particles)
{
    std::tie(valid_likelihood_, likelihood_) = likelihood_model_->likelihood(*measurement_model_, pred_particles.state());

    cor_particles = pred_particles;

    if (valid_likelihood_)
        cor_particles.weight() += (likelihood_.array() + std::numeric_limits<double>::min()).log().matrix();
}


std::pair<bool, VectorXd> BoostrapCorrection::getLikelihood()
{
    return std::make_pair(valid_likelihood_, likelihood_);
}


void BoostrapCorrection::setLikelihoodModel(std::unique_ptr<LikelihoodModel> likelihood_model)
{
    likelihood_model_ = std::move(likelihood_model);
}


void BoostrapCorrection::setMeasurementModel(std::unique_ptr<MeasurementModel> measurement_model)
{
    measurement_model_ = std::move(measurement_model);
}


LikelihoodModel& BoostrapCorrection::getLikelihoodModel()
{
    return *likelihood_model_;
}


MeasurementModel& BoostrapCorrection::getMeasurementModel()
{
    return *measurement_model_;
}
