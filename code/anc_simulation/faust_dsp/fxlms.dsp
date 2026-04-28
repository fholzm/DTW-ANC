declare name "ANC control filter";
declare author "Felix Holzmüller";
declare copyright "IEM";
declare version "20260413_2200";
declare license "MIT";

declare options "[osc:on]";

import("stdfaust.lib");

fxlms_2e2u(order, ctl_stepsize, ctl_reset, ctl_adapt, ctl_delay, e0, e1, x, r00, r01, r10, r11) = (si.bus(2 * order) ~ adaptation) : apply_filter : _, _  with {

    adaptation = si.bus(2*order) <:
                 par(i, order,
                     ba.selector(i, 2*order) :
                     - (ctl_adapt * ctl_stepsize * (e0 * (r00 : @(i + ctl_delay)) +
                                                e1 * (r10 : @(i + ctl_delay))))),
                 par(i, order,
                     ba.selector(i+order, 2*order) :
                     - (ctl_adapt * ctl_stepsize * (e0 * (r01 : @(i + ctl_delay)) +
                                                e1 * (r11 : @(i + ctl_delay))))) :
                 par(i, order*2, *(ctl_reset * -1 + 1));

    apply_filter = si.bus(2*order) <:
                   sum(i, order,
                       ba.selector(i, 2*order) * (x : @(i))),
                   sum(i, order, ba.selector(i+order, 2*order) * (x : @(i)));
};


order = 256;
in_level = abs : ba.linear2db : si.smoo : hbargraph("[7]Level L[unit:dB]",-100,-40);


process = (_ <: attach(_, in_level)), si.bus(6): vgroup("Adaptation", fxlms_2e2u(order, ctl_stepsize, ctl_reset, ctl_adapt, ctl_delay)) : _, _
with{
    ctl_stepsize = hslider("[4]Stepsize", 0, 0, 10, 0.0001);
    ctl_reset = button("[2]Reset");
    ctl_adapt = checkbox("[1]Disable adaptation") *(-1) +(1);
    ctl_delay = nentry("[6]Delay", 0, 0, 100, 1);
};
