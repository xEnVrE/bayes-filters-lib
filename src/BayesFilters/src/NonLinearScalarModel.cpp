
#include <BayesFilters/NonLinearScalarModel.h>

#include <cmath>
#include <utility>

#include <Eigen/Cholesky>
#include <iostream>
using namespace bfl;
using namespace Eigen;

constexpr int STATE_SIZE = 1;

NonLinearScalarModel::NonLinearScalarModel
(
    double variance,
    unsigned int seed
) noexcept :
    generator_(std::mt19937_64(seed)),
    distribution_(std::normal_distribution<double>(0.0, 1.0)),
    Q_(STATE_SIZE, STATE_SIZE),
    gauss_rnd_sample_([&] { return (distribution_)(generator_); })
{
    Q_(0, 0) = variance;
}


NonLinearScalarModel::NonLinearScalarModel(double variance) noexcept :
    NonLinearScalarModel(variance, 1)
{ }


NonLinearScalarModel::NonLinearScalarModel() noexcept :
    NonLinearScalarModel(1.0, 1)
{ }


NonLinearScalarModel::~NonLinearScalarModel() noexcept
{ }


NonLinearScalarModel::NonLinearScalarModel(const NonLinearScalarModel& model) :
    generator_(model.generator_),
    distribution_(model.distribution_),
    Q_(model.Q_),
    gauss_rnd_sample_(model.gauss_rnd_sample_)
{ }


NonLinearScalarModel::NonLinearScalarModel(NonLinearScalarModel&& model) noexcept :
    generator_(std::move(model.generator_)),
    distribution_(std::move(model.distribution_)),
    gauss_rnd_sample_(std::move(model.gauss_rnd_sample_))
{ }


NonLinearScalarModel& NonLinearScalarModel::operator=(const NonLinearScalarModel& model)
{
    NonLinearScalarModel tmp(model);
    *this = std::move(tmp);

    return *this;
}


NonLinearScalarModel& NonLinearScalarModel::operator=(NonLinearScalarModel&& model) noexcept
{
    Q_ = std::move(model.Q_);

    generator_        = std::move(model.generator_);
    distribution_     = std::move(model.distribution_);
    gauss_rnd_sample_ = std::move(model.gauss_rnd_sample_);

    return *this;
}

void NonLinearScalarModel::propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> pred_states)
{
    // see example 15.1 in book Optimal State Estimation Kalman, H-infinity, and Nonlinear Approaches
    pred_states = (0.5 * cur_states.array() + 25 * cur_states.array() * (1 + cur_states.array().pow(2)).inverse()).matrix();
}

MatrixXd NonLinearScalarModel::getNoiseSample(const std::size_t num)
{
    MatrixXd rand_vectors(getOutputSize().first, num);
    for (int i = 0; i < rand_vectors.size(); i++)
        *(rand_vectors.data() + i) = gauss_rnd_sample_();

    return std::sqrt(Q_(0, 0)) * rand_vectors;
}


MatrixXd NonLinearScalarModel::getNoiseCovarianceMatrix()
{
    return Q_;
}

VectorXd NonLinearScalarModel::getTransitionProbability(const Ref<const MatrixXd>& prev_states, Ref<MatrixXd> cur_states)
{
    VectorXd probabilities(prev_states.cols());
    MatrixXd differences = cur_states - prev_states;

    std::size_t size = differences.rows();
    for (std::size_t i = 0; i < prev_states.cols(); i++)
    {
        probabilities(i) = (-0.5 * static_cast<double>(size) * log(2.0 * M_PI) + -0.5 * log(Q_.determinant()) -0.5 * (differences.col(i).transpose() * Q_.inverse() * differences.col(i)).array()).exp().coeff(0);
    }

    return probabilities;
}


std::pair<std::size_t, std::size_t> NonLinearScalarModel::getOutputSize() const
{
    return std::make_pair(STATE_SIZE, 0);
}
