#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// TODO: template-based implementation to support double precision directly

// Example impulse responses for testing
static const std::vector<std::vector<double>> irs_clean = { { 1.0, 1.0, 0.0, 0.0, 0.0 },
                                                            { 1.0, 0.0, 1.0, 0.0, 0.0 },
                                                            { 1.0, 0.0, 0.0, 1.0, 0.0 },
                                                            { 1.0, 0.0, 0.0, 0.0, 1.0 } };

// Positions corresponding to the impulse responses
static const std::vector<double> ir_positions = { 0.0, 0.33, 0.66, 1.0 };

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