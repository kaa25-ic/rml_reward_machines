:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).

match(_event, a_match(N)) :- deep_subdict(_event, _{'a':N}), >(N, 0).
match(_event, b_match) :- deep_subdict(_event, _{'b':T}), T=1.0.
match(_event, c_match) :- deep_subdict(_event, _{'c':T}), T=1.0.
match(_event, d_match) :- deep_subdict(_event, _{'d':T}), T=1.0.
match(_event, not_abcd) :-
    not(match(_event, a_match(_))),
    not(match(_event, b_match)),
    not(match(_event, c_match)),
    not(match(_event, d_match)).
match(_, any).

trace_expression('Main', Main) :-
    Main = (star((not_abcd:eps)) * var(n, ((a_match(var(n)):eps) * app(S0_B, [var('n')])))),
    S0_B = gen(['n'], (star((not_abcd:eps)) * ((b_match:eps) * app(S1_D, [var('n')])))),
    S1_D = gen(['n'], (star((not_abcd:eps)) * ((d_match:eps) * app(S2_C, [var('n')])))),
    S2_C = gen(['n'], (star((not_abcd:eps)) * ((c_match:eps) * app(S3_D, [var('n')])))),
    S3_D = gen(['n'], guarded((var('n') > 0), (star((not_abcd:eps)) * ((d_match:eps) * app(S3_D, [(var('n') - 1)]))), 1)).
