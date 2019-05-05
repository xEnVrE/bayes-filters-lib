/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/LinearModel.h>

#include <cmath>
#include <iostream>
#include <utility>

using namespace bfl;
using namespace Eigen;


LinearModel::LinearModel
(
    const double sigma_x,
    const double sigma_y,
    const unsigned int seed
) noexcept :
    generator_(std::mt19937_64(seed)),
    distribution_(std::normal_distribution<double>(0.0, 1.0)),
    gauss_rnd_sample_([&] { return (distribution_)(generator_); })
{
    H_.resize(1, 1);
    H_(0, 0) = 1.0;

    R_.resize(1, 1);
    R_(0, 0) = std::pow(sigma_x, 2.0);
}


LinearModel::LinearModel(const double sigma_x, const double sigma_y) noexcept :
    LinearModel(sigma_x, sigma_y, 1) { }


LinearModel::LinearModel() noexcept :
    LinearModel(10.0, 10.0, 1) { }


LinearModel::~LinearModel() noexcept
{ }


LinearModel::LinearModel(const LinearModel& lin_sense) :
    H_(lin_sense.H_),
    R_(lin_sense.R_),
    gauss_rnd_sample_(lin_sense.gauss_rnd_sample_)
{
    if (lin_sense.log_enabled_)
    {
        std::cerr << "WARNING::LINEARSENSOR::OPERATOR=\n";
        std::cerr << "\tWARNING: Source object has log enabled, but log file stream cannot be copied. Use target object enableLog(const std::string&) to enable logging." << std::endl;
    }
}


LinearModel::LinearModel(LinearModel&& lin_sense) noexcept :
    generator_(std::move(lin_sense.generator_)),
    distribution_(std::move(lin_sense.distribution_)),
    H_(std::move(lin_sense.H_)),
    R_(std::move(lin_sense.R_)),
    gauss_rnd_sample_(std::move(lin_sense.gauss_rnd_sample_))
{
    if (lin_sense.log_enabled_)
    {
        lin_sense.disable_log();

        enable_log(lin_sense.get_prefix_path(), lin_sense.get_prefix_name());
    }
}


LinearModel& LinearModel::operator=(const LinearModel& lin_sense) noexcept
{
    H_ = lin_sense.H_;
    R_ = lin_sense.R_;

    generator_ = lin_sense.generator_;
    distribution_ = lin_sense.distribution_;
    gauss_rnd_sample_ = lin_sense.gauss_rnd_sample_;

    if (lin_sense.log_enabled_)
    {
        std::cerr << "WARNING::LINEARSENSOR::OPERATOR=\n";
        std::cerr << "\tWARNING: Source object has log enabled, but log file stream cannot be copied. Use target object enableLog(const std::string&) to enable logging." << std::endl;
    }

    return *this;
}


LinearModel& LinearModel::operator=(LinearModel&& lin_sense) noexcept
{
    H_       = std::move(lin_sense.H_);
    R_       = std::move(lin_sense.R_);

    generator_        = std::move(lin_sense.generator_);
    distribution_     = std::move(lin_sense.distribution_);
    gauss_rnd_sample_ = std::move(lin_sense.gauss_rnd_sample_);

    if (lin_sense.log_enabled_)
    {
        lin_sense.disable_log();

        enable_log(lin_sense.get_prefix_path(), lin_sense.get_prefix_name());
    }

    return *this;
}


std::pair<bool, MatrixXd> LinearModel::getNoiseSample(const int num) const
{
    MatrixXd rand_vectors(2, num);
    for (int i = 0; i < rand_vectors.size(); i++)
        *(rand_vectors.data() + i) = gauss_rnd_sample_();

    MatrixXd noise_sample = std::sqrt(R_(0, 0)) * rand_vectors;

    return std::make_pair(true, std::move(noise_sample));
}


std::pair<bool, MatrixXd> LinearModel::getNoiseCovarianceMatrix() const
{
    return std::make_pair(true, R_);
}


Eigen::MatrixXd LinearModel::getMeasurementMatrix() const
{
    return H_;
}
