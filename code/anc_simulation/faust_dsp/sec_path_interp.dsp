declare name "Secondary path interpolator";
declare author "Felix Holzmüller";
declare copyright "IEM";
declare version "20260417_1000";
declare license "GPLv3";
declare options "[osc:on]";

import("stdfaust.lib");

IR_SIZE = 128;

// Interpolator functions from interpolator.h
interpolate_c = ffunction( float interpolate(float, int, int, int), "interpolator.h", "");
get_alpha_c = ffunction( float get_alpha(float), "interpolator.h", "");

// Controlable position
position = hslider("[1]Position", 0.5, 0.5, 1.1, 0.001) : si.smoo;

// Get interpolation factor alpha based on position
alpha = get_alpha_c(position);

// Get interpolated filter coefficients, based on selected method
interpolation_method = nentry("[0]Method[style:menu{'NN':0;'Linear':1;'Global Alignment':2;'DTW':3}]", 0, 0, 3, 1);
fir_coeffs_0_0 = par(i, IR_SIZE, interpolate_c(alpha, i, interpolation_method, 0));
fir_coeffs_0_1 = par(i, IR_SIZE, interpolate_c(alpha, i, interpolation_method, 1));

// Process audio with FIR filters
process = _ <: fi.fir(fir_coeffs_0_0), fi.fir(fir_coeffs_0_1);
