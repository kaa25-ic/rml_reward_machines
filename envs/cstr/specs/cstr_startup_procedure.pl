:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).

match(_event, critical) :-
    deep_subdict(_event, _{'critical':T}), T=1.0.
match(_event, unsafe) :-
    deep_subdict(_event, _{'temp_safe':T}), T=0.0,
    not(match(_event, critical)).
match(_event, done_regulated) :-
    deep_subdict(_event, _{'terminate':T}), T=true,
    deep_subdict(_event, _{'stable':S}), S=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)).
match(_event, done_unregulated) :-
    deep_subdict(_event, _{'terminate':T}), T=true,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)).
match(_event, deadline) :-
    deep_subdict(_event, _{'past_deadline':T}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)).
match(_event, overshoot) :-
    deep_subdict(_event, _{'overshoot':T}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)).
match(_event, stable) :-
    deep_subdict(_event, _{'stable':T}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)).
match(_event, in_soak) :-
    deep_subdict(_event, _{'in_soak_band':T}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)).
match(_event, safe) :-
    deep_subdict(_event, _{'temp_safe':T}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)),
    not(match(_event, in_soak)).
match(_event, other) :-
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)),
    not(match(_event, in_soak)),
    not(match(_event, safe)).
match(_, any).

trace_expression('Main', Main) :-
    Main = app(Preheat, []),
    Preheat = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_1, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_1 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_2, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_2 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_3, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_3 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_4, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_4 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_5, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_5 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_6, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_6 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_7, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_7 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_8, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_8 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_9, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_9 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:0)
        \/ (in_soak:eps) * app(Soak_10, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Soak_10 = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:eps) * app(Regulate, [])
        \/ (in_soak:eps) * app(Approach, [])
        \/ (safe:eps) * app(Preheat, [])
        \/ (other:0)
    )),
    Approach = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:0)
        \/ (done_unregulated:0)
        \/ (deadline:0)
        \/ (overshoot:0)
        \/ (stable:eps) * app(Regulate, [])
        \/ (in_soak:eps) * app(Approach, [])
        \/ (safe:eps) * app(Approach, [])
        \/ (other:0)
    )),
    Regulate = gen([], (
        (critical:0)
        \/ (unsafe:0)
        \/ (done_regulated:1)
        \/ (done_unregulated:0)
        \/ (deadline:eps) * app(Regulate, [])
        \/ (overshoot:0)
        \/ (stable:eps) * app(Regulate, [])
        \/ (in_soak:eps) * app(Regulate, [])
        \/ (safe:0)
        \/ (other:0)
    )).
