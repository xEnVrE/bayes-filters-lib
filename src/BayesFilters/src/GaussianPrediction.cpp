#include <BayesFilters/GaussianPrediction.h>

#include <exception>
#include <iostream>

using namespace bfl;
using namespace Eigen;


GaussianPrediction::GaussianPrediction() noexcept { };


GaussianPrediction::GaussianPrediction(GaussianPrediction&& g_prediction) noexcept { }


void GaussianPrediction::predict(const GaussianMixture& prev_state, GaussianMixture& pred_state)
{
    if (!skip_prediction_)
        predictStep(prev_state, pred_state);
    else
        pred_state = prev_state;
}


bool GaussianPrediction::skip(const std::string& what_step, const bool status)
{
    if (what_step == "prediction")
        skip_prediction_ = status;
    else if (what_step == "state")
        skip_state_ = status;
    else if (what_step == "exogenous")
        skip_exogenous_ = status;
    else
        return false;

    return true;
}


bool GaussianPrediction::getSkipState()
{
    return skip_state_;
}


bool GaussianPrediction::getSkipExogenous()
{
    return skip_exogenous_;
}

