/*
==============================================================================
Time-aligned secondary path interpolation for ANC
Copyright (C) 2026  Felix Holzmüller <holzmueller@iem.at>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
==============================================================================
*/

#include "lib/spline.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "eval_anc_irs.h"

// TODO: template-based implementation to support double precision directly
// Initialize spline object
tk::spline s (ir_index_warped, ir_interpolated_warped);

double last_alpha = -1.0;

double get_alpha (double position)
{
    // Calculate interpolation index, where the decimals refer to alpha and the
    // integer part to the index of the lower IR

    // Handle boundary conditions
    if (position <= ir_positions[0])
    {
        return 0.0;
    }

    // Find surrounding positions for interpolation
    for (size_t i = 1; i < ir_positions.size(); ++i)
    {
        // Return clean value if position matches exactly
        if (position == ir_positions[i])
        {
            return static_cast<double> (i);
        }

        // Check if position is in this interval
        if (position > ir_positions[i - 1] && position < ir_positions[i])
        {
            double alpha =
                (position - ir_positions[i - 1]) / (ir_positions[i] - ir_positions[i - 1]);
            return static_cast<double> (i - 1) + alpha;
        }
    }

    return static_cast<double> (ir_positions.size() - 1);
}

double interpolate_direct (double alpha, int index)
{
    // Sanitize input index
    index = std::max (0, std::min (index, static_cast<int> (irs_clean[0].size() - 1)));

    // Extract IR indices and alpha for interpolation
    int upper_ir_idx =
        std::min (static_cast<int> (alpha) + 1, static_cast<int> (ir_positions.size()) - 1);
    int lower_ir_idx = std::max (0, std::min (static_cast<int> (alpha), upper_ir_idx));

    alpha -= static_cast<double> (lower_ir_idx);

    return ((1.0 - alpha) * irs_clean[lower_ir_idx][index]
            + alpha * irs_clean[upper_ir_idx][index]);
}

double interpolate_nn (double alpha, int index)
{
    // Nearest neighbor interpolation
    int ir_idx = static_cast<int> (std::round (alpha));
    ir_idx = std::max (0, std::min (ir_idx, static_cast<int> (ir_positions.size() - 1)));

    // Sanitize input index
    index = std::max (0, std::min (index, static_cast<int> (irs_clean[0].size() - 1)));

    return irs_clean[ir_idx][index];
}

double interpolate_dtw (double alpha, int index)
{
    // Sanitize input index
    index = std::max (0, std::min (index, static_cast<int> (irs_clean[0].size() - 1)));

    // Extract IR indices and alpha for interpolation
    int upper_ir_idx =
        std::min (static_cast<int> (alpha) + 1, static_cast<int> (ir_positions.size()) - 1);
    int lower_ir_idx = std::max (0, std::min (static_cast<int> (alpha), upper_ir_idx));

    double alpha_dec = alpha - static_cast<double> (lower_ir_idx);

    // Interpolate only once at first call for index 0
    if (index == 0 && alpha != last_alpha)
    {
        last_alpha = alpha;

        // Select which warped IR to use
        const std::vector<double>* ir_pos0;
        const std::vector<double>* ir_pos1;
        const std::vector<double>* displacement;

        if (alpha_dec < 0.5)
        {
            ir_pos0 = &irs_clean[lower_ir_idx];
            ir_pos1 = &irs_upper_warped[upper_ir_idx];
            displacement = &displacement_upper[upper_ir_idx];
        }
        else
        {
            ir_pos0 = &irs_lower_warped[lower_ir_idx];
            ir_pos1 = &irs_clean[upper_ir_idx];
            displacement = &displacement_lower[lower_ir_idx];
        }

        // Linear interpolate original and warped IRs
        for (size_t n = 0; n < ir_interpolated_warped.size(); ++n)
        {
            ir_interpolated_warped[n] =
                (1.0 - alpha_dec) * (*ir_pos0)[n] + alpha_dec * (*ir_pos1)[n];

            // // Calculate filter tap indices for dewarping
            ir_index_warped[n] =
                static_cast<double> (n)
                - ((*displacement)[n] * ((alpha_dec < 0.5) ? alpha_dec : (1.0 - alpha_dec)));
        }

        // Update spline
        s = tk::spline (ir_index_warped,
                        ir_interpolated_warped,
                        tk::spline::cspline,
                        false,
                        tk::spline::first_deriv,
                        0.0,
                        tk::spline::first_deriv,
                        0.0);
    }

    // Evaluate spline at given index
    return s (static_cast<double> (index));
}

double interpolate (double alpha, int index, int method)
{
    switch (method)
    {
        case 0:
            return interpolate_nn (alpha, index);
        case 1:
            return interpolate_direct (alpha, index);
        case 2:
            return interpolate_dtw (alpha, index);
        default:
            return interpolate_nn (alpha, index);
    }
}