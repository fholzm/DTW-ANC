declare name "Secondary path interpolator";
declare author "Felix Holzmüller";
declare copyright "IEM";
declare version "20260128_1700";
declare license "GPLv3";

declare options "[osc:on]";

import("stdfaust.lib");

IR_SIZE = 64;

// interpolate_c = ffunction( float interpolate_nn(float, int), "interpolator.h", "");
interpolate_c = ffunction( float interpolate(float, int, int), "interpolator.h", "");
get_alpha_c = ffunction( float get_alpha(float), "interpolator.h", "");

position = hslider("[1]Position", 0.25, 0.25, 1.0, 0.001) : si.smoo;

alpha = get_alpha_c(position);
// index = hslider("index", 0, 0, IR_SIZE - 1, 1);

interpolation_method = nentry("[0]Method[style:menu{'NN':0;'Linear':1;'DTW':2}]", 0, 0, 2, 1);
fir_coeffs = par(i, IR_SIZE, interpolate_c(alpha, i, interpolation_method));

/* Audio-rate usage */
process = _ : fi.fir(fir_coeffs) <: attach(_, hbargraph("Output", 0, 1));
