declare name "ANC control filter";
declare author "Felix Holzmüller";
declare copyright "IEM";
declare version "20240618_1650";
declare license "MIT";

declare options "[osc:on]";

import("stdfaust.lib");

fxlms_siso(order, ctl_stepsize, ctl_reset, ctl_adapt, ctl_delay, e, x, r) = (si.bus(order) ~ adaptation) : apply_filter : _  with {

    adaptation = si.bus(order) <: par(i, order, ba.selector(i, order) : - (stepsize * ctl_adapt * e * (r : @(i + ctl_delay)))) : par(i, order, *(ctl_reset * -1 + 1));
    apply_filter = si.bus(order) <: sum(i, order, ba.selector(i, order) * (x : @(i)));
};

order = 128;
in_level = abs : ba.linear2db : si.smoo : hbargraph("[7]Level L[unit:dB]",-100,-40);


process = (_ <: attach(_, in_level)), si.bus(2): vgroup("Adaptation", fxlms_siso(order, ctl_stepsize, ctl_reset, ctl_adapt, ctl_delay)) : _
with{
    ctl_stepsize = hslider("[4]Stepsize", 0, 0, 10, 0.0001);
    ctl_reset = button("[2]Reset");
    ctl_adapt = checkbox("[1]Disable adaptation") *(-1) +(1);
    ctl_delay = nentry("[6]Delay", 0, 0, 100, 1);
};
