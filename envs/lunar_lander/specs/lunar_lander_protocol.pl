:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).

match(_event, corridor_match) :-
    deep_subdict(_event, _{'corridor':T}),
    T=1.0.

match(_event, hover_match) :-
    deep_subdict(_event, _{'hover':T}),
    T=1.0.

match(_event, controlled_descent_match) :-
    deep_subdict(_event, _{'controlled_descent':T}),
    T=1.0.

match(_event, safe_landing_match) :-
    deep_subdict(_event, _{'both_contact':B}),
    deep_subdict(_event, _{'target_zone':Z}),
    deep_subdict(_event, _{'safe_landing_angle':A}),
    deep_subdict(_event, _{'env_terminated':T}),
    deep_subdict(_event, _{'env_successful_landing':S}),
    B=1.0,
    Z=1.0,
    A=1.0,
    T=1.0,
    S=1.0.

match(_event, episode_ended) :-
    deep_subdict(_event, _{'env_terminated':T}),
    T=1.0.
match(_event, episode_ended) :-
    deep_subdict(_event, _{'env_truncated':T}),
    T=1.0.

match(_event, waiting_for_corridor) :-
    not(match(_event, corridor_match)),
    not(match(_event, episode_ended)).

match(_event, waiting_for_hover) :-
    not(match(_event, hover_match)),
    not(match(_event, episode_ended)).

match(_event, waiting_for_descent) :-
    not(match(_event, controlled_descent_match)),
    not(match(_event, episode_ended)).

match(_event, waiting_for_landing) :-
    not(match(_event, safe_landing_match)),
    not(match(_event, episode_ended)).

trace_expression('Main', Main) :-
    Main = (
        star((waiting_for_corridor:eps)) *
        (corridor_match:eps) *
        app(Hover, [0])
    ),
    Hover = gen(
        ['h'],
        guarded(
            (var('h') < 2),
            (star((waiting_for_hover:eps)) * ((hover_match:eps) * app(Hover, [(var('h') + 1)]))),
            app(HoverComplete, [])
        )
    ),
    HoverComplete = gen(
        [],
        (
            star((waiting_for_hover:eps)) *
            (hover_match:eps) *
            app(ControlledDescent, [])
        )
    ),
    ControlledDescent = gen(
        [],
        (
            star((waiting_for_descent:eps)) *
            (controlled_descent_match:eps) *
            star((waiting_for_landing:eps)) *
            (safe_landing_match:eps) *
            1
        )
    ).
