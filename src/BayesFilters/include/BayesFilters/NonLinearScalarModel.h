
#pragma once

#include <BayesFilters/AdditiveStateModel.h>

#include <random>


namespace bfl {
    class NonLinearScalarModel;
}


class bfl::NonLinearScalarModel : public AdditiveStateModel
{
public:
    NonLinearScalarModel(double T, double tilde_q, unsigned int seed) noexcept;

    NonLinearScalarModel(double T, double tilde_q) noexcept;

    NonLinearScalarModel() noexcept;

    NonLinearScalarModel(const NonLinearScalarModel& wna);

    NonLinearScalarModel(NonLinearScalarModel&& wna) noexcept;

    virtual ~NonLinearScalarModel() noexcept;

    NonLinearScalarModel& operator=(const NonLinearScalarModel& wna);

    NonLinearScalarModel& operator=(NonLinearScalarModel&& wna) noexcept;

    void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> prop_states) override;

    Eigen::MatrixXd getNoiseSample(const std::size_t num) override;

    Eigen::MatrixXd getNoiseCovarianceMatrix() override;

    Eigen::VectorXd getTransitionProbability(const Eigen::Ref<const Eigen::MatrixXd>& prev_states, Eigen::Ref<Eigen::MatrixXd> cur_states);

    bool setProperty(const std::string& property) override { return false; };

    std::pair<std::size_t, std::size_t> getOutputSize() const override;

private:
    std::mt19937_64 generator_;

    std::normal_distribution<double> distribution_;

protected:
    /**
     * Sampling interval in [time].
     */
    double T_;

    /**
     * Convariance matrix of the additive white noise of the state model.
     */
    Eigen::MatrixXd Q_;

    /**
     * Power spectral density [length]^2/[time]^3.
     */
    double tilde_q_;

    /**
     * Square root matrix of R_.
     */
    Eigen::MatrixXd sqrt_Q_;

    /**
     * Random number generator function from a Normal distribution.
     * A call to `gauss_rnd_sample_()` returns a double-precision floating point random number.
     */
    std::function<double()> gauss_rnd_sample_;
};
