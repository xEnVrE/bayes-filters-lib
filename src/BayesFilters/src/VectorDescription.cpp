/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/VectorDescription.h>

using namespace bfl;


VectorDescription::VectorDescription
(
    const std::size_t linear_components,
    const std::size_t circular_components,
    const std::size_t noise_components,
    const VectorDescription::CircularType& circular_type
) :
    total_size(linear_components + circular_components + noise_components),
    linear_size(linear_components),
    circular_size(circular_components),
    noise_size(noise_components),
    linear_components(linear_components),
    circular_components(circular_components),
    noise_components(noise_components),
    circular_type(circular_type)
{
    if (circular_type == CircularType::Quaternion)
    {
        circular_size = circular_components * 4;
        total_size = linear_size + circular_size + noise_size;
    }
}
