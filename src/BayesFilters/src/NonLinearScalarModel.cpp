
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
    double T,
    double tilde_q,
    unsigned int seed
) noexcept :
    generator_(std::mt19937_64(seed)),
    distribution_(std::normal_distribution<double>(0.0, 1.0)),
    T_(T),
    Q_(STATE_SIZE, STATE_SIZE),
    tilde_q_(tilde_q),
    sqrt_Q_(STATE_SIZE, STATE_SIZE),
    gauss_rnd_sample_([&] { return (distribution_)(generator_); })
{
    double q11 = 1.0/3.0 * std::pow(T_, 3.0);
    Q_ << q11;
    Q_ *= tilde_q;

    LDLT<MatrixXd> chol_ldlt(Q_);
    MatrixXd tmpIdentity(STATE_SIZE, STATE_SIZE);
    tmpIdentity.setIdentity();
    sqrt_Q_ = (chol_ldlt.transpositionsP() * tmpIdentity).transpose() * chol_ldlt.matrixL() * chol_ldlt.vectorD().real().cwiseSqrt().asDiagonal();
}


NonLinearScalarModel::NonLinearScalarModel(double T, double tilde_q) noexcept :
    NonLinearScalarModel(T, tilde_q, 1)
{ }


NonLinearScalarModel::NonLinearScalarModel() noexcept :
    NonLinearScalarModel(1.0, 1.0, 1)
{ }


NonLinearScalarModel::~NonLinearScalarModel() noexcept
{ }


NonLinearScalarModel::NonLinearScalarModel(const NonLinearScalarModel& wna) :
    generator_(wna.generator_),
    distribution_(wna.distribution_),
    T_(wna.T_),
    Q_(wna.Q_),
    tilde_q_(wna.tilde_q_),
    sqrt_Q_(wna.sqrt_Q_),
    gauss_rnd_sample_(wna.gauss_rnd_sample_)
{ }


NonLinearScalarModel::NonLinearScalarModel(NonLinearScalarModel&& wna) noexcept :
    generator_(std::move(wna.generator_)),
    distribution_(std::move(wna.distribution_)),
    T_(wna.T_),
    Q_(std::move(wna.Q_)),
    tilde_q_(wna.tilde_q_),
    sqrt_Q_(std::move(wna.sqrt_Q_)),
    gauss_rnd_sample_(std::move(wna.gauss_rnd_sample_))
{
    wna.T_       = 0.0;
    wna.tilde_q_ = 0.0;
}


NonLinearScalarModel& NonLinearScalarModel::operator=(const NonLinearScalarModel& wna)
{
    NonLinearScalarModel tmp(wna);
    *this = std::move(tmp);

    return *this;
}


NonLinearScalarModel& NonLinearScalarModel::operator=(NonLinearScalarModel&& wna) noexcept
{
    T_       = wna.T_;
    Q_       = std::move(wna.Q_);
    tilde_q_ = wna.tilde_q_;

    sqrt_Q_           = std::move(wna.sqrt_Q_);
    generator_        = std::move(wna.generator_);
    distribution_     = std::move(wna.distribution_);
    gauss_rnd_sample_ = std::move(wna.gauss_rnd_sample_);

    wna.T_       = 0.0;
    wna.tilde_q_ = 0.0;

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

    return sqrt_Q_ * rand_vectors;
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
