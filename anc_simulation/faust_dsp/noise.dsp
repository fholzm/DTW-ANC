declare name "Noise Generator";
declare author "Felix Holzmüller";
declare copyright "IEM";
declare version "20260128_1630";
declare license "GPLv3";

declare options "[osc:on]";

import("stdfaust.lib");

// Configs for noise generator
n_noises = 100;
noise_index = nentry("[3]Realization", 0, 0, n_noises-1, 1);

// Configs for LP-filter
fc = hslider("[2]LP[unit:Hz][scale:log]", 2000, 20, 20000, 0.1);
N = 6;

process = no.multinoise(n_noises) : ba.selectn(n_noises, noise_index) : fi.lowpass(N, fc) : *(active) with{
    active = checkbox("[1]Active");
};
