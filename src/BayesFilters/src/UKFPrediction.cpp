#include <BayesFilters/UKFPrediction.h>

using namespace bfl;
using namespace bfl::sigma_point;
using namespace Eigen;


UKFPrediction::UKFPrediction
(
    std::unique_ptr<StateModel> state_model,
    const size_t n,
    const double alpha,
    const double beta,
    const double kappa
) noexcept :
    state_model_(std::move(state_model)),
    type_(UKFPredictionType::Generic),
    ut_weight_(n, alpha, beta, kappa)
{ }


UKFPrediction::UKFPrediction
(
    std::unique_ptr<AdditiveStateModel> state_model,
    const size_t n,
    const double alpha,
    const double beta,
    const double kappa
) noexcept :
    add_state_model_(std::move(state_model)),
    type_(UKFPredictionType::Additive),
    ut_weight_(n, alpha, beta, kappa)
{ }


UKFPrediction::UKFPrediction(UKFPrediction&& ukf_prediction) noexcept:
    state_model_(std::move(ukf_prediction.state_model_)),
    add_state_model_(std::move(ukf_prediction.add_state_model_)),
    type_(ukf_prediction.type_),
    ut_weight_(ukf_prediction.ut_weight_)
{ }


UKFPrediction::~UKFPrediction() noexcept
{ }


void UKFPrediction::setExogenousModel(std::unique_ptr<ExogenousModel> exog_model)
{
    exog_model_ = std::move(exog_model);
}


void UKFPrediction::predictStep(const GaussianMixture& prev_state, GaussianMixture& pred_state)
{
    bool skip_exogenous = getSkipExogenous() || (exog_model_ == nullptr);
    if ( getSkipState() && skip_exogenous )
    {
        /* Skip prediction step entirely. */
        pred_state = prev_state;
        return;
    }

    if ((!getSkipState()) && (!skip_exogenous))
    {
        /* Evaluate predicted mean and predicted covariance using unscented transform. */
        if (type_ == UKFPredictionType::Generic)
        {
            /* Augment the previous state using process noise statistics. */
            GaussianMixture prev_state_augmented = prev_state;
            prev_state_augmented.augmentWithNoise(state_model_->getNoiseCovarianceMatrix());

            std::tie(pred_state, std::ignore) = unscented_transform(prev_state_augmented, ut_weight_, *state_model_, *exog_model_);
        }
        else if (type_ == UKFPredictionType::Additive)
        {
            std::tie(pred_state, std::ignore) = unscented_transform(prev_state, ut_weight_, *add_state_model_, *exog_model_);
        }
    }
    else if (!getSkipState())
    {
        /* Evaluate predicted mean and predicted covariance using unscented transform. */
        if (type_ == UKFPredictionType::Generic)
        {
            /* Augment the previous state using process noise statistics. */
            GaussianMixture prev_state_augmented = prev_state;
            prev_state_augmented.augmentWithNoise(state_model_->getNoiseCovarianceMatrix());

            std::tie(pred_state, std::ignore) = unscented_transform(prev_state_augmented, ut_weight_, *state_model_);
        }
        else if (type_ == UKFPredictionType::Additive)
        {
            std::tie(pred_state, std::ignore) = unscented_transform(prev_state, ut_weight_, *add_state_model_);
        }
    }
    else if (!skip_exogenous)
        std::tie(pred_state, std::ignore) = unscented_transform(prev_state, ut_weight_, *exog_model_);
}
