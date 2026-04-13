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

// Initialize spline object
tk::spline s_0_0 (ir_index_warped_0_0, ir_interpolated_warped_0_0);
tk::spline s_0_1 (ir_index_warped_0_1, ir_interpolated_warped_0_1);

double last_alpha = -1.0;
int last_method = -1;

std::vector<double> ir_interpolated_0_0 (irs_clean_0_0[0]);
std::vector<double> ir_interpolated_0_1 (irs_clean_0_1[0]);

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

void interpolate_direct (double alpha)
{
    // Extract IR indices and alpha for interpolation
    int upper_ir_idx =
        std::min (static_cast<int> (alpha) + 1, static_cast<int> (ir_positions.size()) - 1);
    int lower_ir_idx = std::max (0, std::min (static_cast<int> (alpha), upper_ir_idx));

    alpha -= static_cast<double> (lower_ir_idx);

    for (size_t n = 0; n < ir_interpolated_0_0.size(); ++n)
    {
        ir_interpolated_0_0[n] =
            (1.0 - alpha) * irs_clean_0_0[lower_ir_idx][n] + alpha * irs_clean_0_0[upper_ir_idx][n];
        ir_interpolated_0_1[n] =
            (1.0 - alpha) * irs_clean_0_1[lower_ir_idx][n] + alpha * irs_clean_0_1[upper_ir_idx][n];
    }
}

void interpolate_nn (double alpha)
{
    // Nearest neighbor interpolation
    int ir_idx = static_cast<int> (std::round (alpha));
    ir_idx = std::max (0, std::min (ir_idx, static_cast<int> (ir_positions.size() - 1)));

    // Store in interpolated IR vector
    for (size_t n = 0; n < ir_interpolated_0_0.size(); ++n)
    {
        ir_interpolated_0_0[n] = irs_clean_0_0[ir_idx][n];
        ir_interpolated_0_1[n] = irs_clean_0_1[ir_idx][n];
    }
}

void apply_shift (std::vector<double>& ir, int shift)
{
    if (shift < 0)
    {
        // Shift left: remove from beginning, append zeros at end
        int left_shift = -shift;
        ir.erase (ir.begin(), ir.begin() + left_shift);
        ir.insert (ir.end(), left_shift, 0.0);
    }
    else if (shift > 0)
    {
        // Shift right: remove from end, prepend zeros at beginning
        ir.erase (ir.end() - shift, ir.end());
        ir.insert (ir.begin(), shift, 0.0);
    }
}

void interpolate_ga (double alpha)
{
    // Extract IR indices and alpha for interpolation
    int upper_ir_idx =
        std::min (static_cast<int> (alpha) + 1, static_cast<int> (ir_positions.size()) - 1);
    int lower_ir_idx = std::max (0, std::min (static_cast<int> (alpha), upper_ir_idx));

    alpha -= static_cast<double> (lower_ir_idx);

    // Shift upper IR according to global alignment
    std::vector<double> ir_shifted_0_0 (irs_clean_0_0[upper_ir_idx]);
    std::vector<double> ir_shifted_0_1 (irs_clean_0_1[upper_ir_idx]);

    int shift_0 = static_cast<int> (global_alignment_0_0[lower_ir_idx]);
    shift_0 = std::max (-static_cast<int> (ir_shifted_0_0.size()) + 1,
                        std::min (shift_0, static_cast<int> (ir_shifted_0_0.size()) - 1));
    int shift_1 = static_cast<int> (global_alignment_0_1[lower_ir_idx]);
    shift_1 = std::max (-static_cast<int> (ir_shifted_0_1.size()) + 1,
                        std::min (shift_1, static_cast<int> (ir_shifted_0_1.size()) - 1));

    static int reconstruction_offset_0 =
        static_cast<int> (std::round (static_cast<double> (-shift_0) * alpha));
    static int reconstruction_offset_1 =
        static_cast<int> (std::round (static_cast<double> (-shift_1) * alpha));

    apply_shift (ir_shifted_0_0, shift_0);
    apply_shift (ir_shifted_0_1, shift_1);

    // Interpolate shifted IR with clean IR
    for (size_t n = 0; n < ir_interpolated_0_0.size(); ++n)
    {
        ir_interpolated_0_0[n] =
            (1.0 - alpha) * irs_clean_0_0[lower_ir_idx][n] + alpha * ir_shifted_0_0[n];
        ir_interpolated_0_1[n] =
            (1.0 - alpha) * irs_clean_0_1[lower_ir_idx][n] + alpha * ir_shifted_0_1[n];
    }

    // Proportional shift back after interpolation
    apply_shift (ir_interpolated_0_0, reconstruction_offset_0);
    apply_shift (ir_interpolated_0_1, reconstruction_offset_1);
}

void interpolate_dtw (double alpha)
{
    // Extract IR indices and alpha for interpolation
    int upper_ir_idx =
        std::min (static_cast<int> (alpha) + 1, static_cast<int> (ir_positions.size()) - 1);
    int lower_ir_idx = std::max (0, std::min (static_cast<int> (alpha), upper_ir_idx));

    double alpha_dec = alpha - static_cast<double> (lower_ir_idx);

    // Select which warped IR to use
    const std::vector<double>* ir_pos0_0_0;
    const std::vector<double>* ir_pos1_0_0;
    const std::vector<double>* displacement_0_0;

    const std::vector<double>* ir_pos0_0_1;
    const std::vector<double>* ir_pos1_0_1;
    const std::vector<double>* displacement_0_1;

    if (alpha_dec < 0.5)
    {
        ir_pos0_0_0 = &irs_clean_0_0[lower_ir_idx];
        ir_pos1_0_0 = &irs_upper_warped_0_0[upper_ir_idx];
        displacement_0_0 = &displacement_upper_0_0[upper_ir_idx];

        ir_pos0_0_1 = &irs_clean_0_1[lower_ir_idx];
        ir_pos1_0_1 = &irs_upper_warped_0_1[upper_ir_idx];
        displacement_0_1 = &displacement_upper_0_1[upper_ir_idx];
    }
    else
    {
        ir_pos0_0_0 = &irs_lower_warped_0_0[lower_ir_idx];
        ir_pos1_0_0 = &irs_clean_0_0[upper_ir_idx];
        displacement_0_0 = &displacement_lower_0_0[lower_ir_idx];

        ir_pos0_0_1 = &irs_lower_warped_0_1[lower_ir_idx];
        ir_pos1_0_1 = &irs_clean_0_1[upper_ir_idx];
        displacement_0_1 = &displacement_lower_0_1[lower_ir_idx];
    }

    // Linear interpolate original and warped IRs
    for (size_t n = 0; n < ir_interpolated_warped_0_0.size(); ++n)
    {
        ir_interpolated_warped_0_0[n] =
            (1.0 - alpha_dec) * (*ir_pos0_0_0)[n] + alpha_dec * (*ir_pos1_0_0)[n];
        ir_interpolated_warped_0_1[n] =
            (1.0 - alpha_dec) * (*ir_pos0_0_1)[n] + alpha_dec * (*ir_pos1_0_1)[n];

        // // Calculate filter tap indices for dewarping
        ir_index_warped_0_0[n] =
            static_cast<double> (n)
            - ((*displacement_0_0)[n] * ((alpha_dec < 0.5) ? alpha_dec : (1.0 - alpha_dec)));
        ir_index_warped_0_1[n] =
            static_cast<double> (n)
            - ((*displacement_0_1)[n] * ((alpha_dec < 0.5) ? alpha_dec : (1.0 - alpha_dec)));
    }

    // Update spline
    s_0_0 = tk::spline (ir_index_warped_0_0,
                        ir_interpolated_warped_0_0,
                        tk::spline::cspline,
                        false,
                        tk::spline::first_deriv,
                        0.0,
                        tk::spline::first_deriv,
                        0.0);

    s_0_1 = tk::spline (ir_index_warped_0_1,
                        ir_interpolated_warped_0_1,
                        tk::spline::cspline,
                        false,
                        tk::spline::first_deriv,
                        0.0,
                        tk::spline::first_deriv,
                        0.0);

    // Evaluate spline and store in interpolated IR vector
    for (size_t n = 0; n < ir_interpolated_0_0.size(); ++n)
    {
        ir_interpolated_0_0[n] = s_0_0 (static_cast<double> (n));
        ir_interpolated_0_1[n] = s_0_1 (static_cast<double> (n));
    }
}

double interpolate (double alpha, int index, int method, int channel)
{
    // Sanitize input index
    index = std::max (0, std::min (index, static_cast<int> (irs_clean_0_0[0].size() - 1)));

    // Sanitize channel
    channel = std::max (0, std::min (channel, 1));

    // Update interpolated IR at index 0 call only if method or alpha changed
    if ((method != last_method || alpha != last_alpha) && index == 0)
    {
        last_method = method;
        last_alpha = alpha;

        switch (method)
        {
            case 0:
                interpolate_nn (alpha);
                break;
            case 1:
                interpolate_direct (alpha);
                break;
            case 2:
                interpolate_ga (alpha);
                break;
            case 3:
                interpolate_dtw (alpha);
                break;
            default:
                interpolate_nn (alpha);
                break;
        }
    }

    if (channel == 0)
        return ir_interpolated_0_0[index];
    else
        return ir_interpolated_0_1[index];
}