import("stdfaust.lib");

IR_SIZE = 64;

// interpolate_c = ffunction( float interpolate_nn(float, int), "interpolator.h", "");
interpolate_c = ffunction( float interpolate_dtw(float, int), "interpolator.h", "");
get_alpha_c = ffunction( float get_alpha(float), "interpolator.h", "");

position = hslider("position", 0.25, 0.25, 1.0, 0.001) : si.smoo;

alpha = get_alpha_c(position);
// index = hslider("index", 0, 0, IR_SIZE - 1, 1);

fir_coeffs = par(i, IR_SIZE, interpolate_c(alpha, i));

/* Audio-rate usage */
process = no.pink_noise * 0.1 : fi.fir(fir_coeffs) <: attach(_, hbargraph("Output", 0, 1));
