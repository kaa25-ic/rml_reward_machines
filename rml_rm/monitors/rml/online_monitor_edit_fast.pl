:- use_module(library(http/websocket)).
:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_client)).
:- use_module(library(http/json)).
:- use_module(library(http/json_convert)).
:- use_module(library(http/http_json)).
:- use_module(library(crypto)).

:- use_module(trace_expressions_semantics).

% Monitor is faster as it does not use print statements


% Using a hash to represent each unique state of the monitor
% Grounds a term by replacing all unbound variables with a consistent representation
ground_term(Term, GroundedTerm) :-
    copy_term(Term, GroundedTerm),
    numbervars(GroundedTerm, 0, _).

% Converts a term to a hashable string after grounding it
term_to_hashable_string(Term, String) :-
    ground_term(Term, GroundedTerm),
    term_string(GroundedTerm, String).

% Initialize the monitor state
initialize_monitor :-
    trace_expression(_, TE),
    nb_setval(state, TE).

:- http_handler(/, http_upgrade_to_websocket(manage_event, []), []). % default options for both the websocket and the http handler

%% the server expects a required first argument: the filename containing the specified trace expression
%% second optional argument: a log file, if not provided no logging is performed

server(Port) :- http_server(http_dispatch, [port('127.0.0.1':Port), workers(1)]). % one worker to guarantee event sequentiality

log(Log) :-   % Used for logging. Helps with things like debugging
    nb_getval(log, Stream), Stream \== null ->  % optional logging of server activity
    (   Log = (TE, E) ->
            writeln(Stream, "Trace expression:"), writeln(Stream, TE), writeln(Stream, "Event: "), writeln(Stream, E);
        Log = (TE, E, error), writeln(Stream, "Trace expression:"), writeln(Stream, TE), writeln(Stream, "Event: "), writeln(Stream, E), writeln(Stream, "Error")),
    nl(Stream),
    flush_output(Stream);
    true.

manage_event(WebSocket) :-
    ws_receive(WebSocket, Msg, [format(json), value_string_as(string)]),
    (   Msg.opcode == close ->      % Logic for receiving a message to close the websocket
            true;
        E = Msg.data,    % E represents content of Websocket message data
        nb_getval(state, TE1),     % Retrieves the value of state and stores it in TE1
        (   next(TE1, E, TE2) ->
            term_to_hashable_string(TE2,Code), 
            nb_setval(state, TE2),
            (   TE2 = 1 ->
                (writeln('verdict = True'), Reply = _{}.put(E).put(_{verdict:true, monitor_state:Code})); % Verdict = True
                (may_halt(TE2) ->
                    (writeln('verdict = ?_True'), Reply = _{}.put(E).put(_{verdict:currently_true, monitor_state:Code})); % Verdict = ?_True
                    (Reply = _{}.put(E).put(_{verdict:currently_false, monitor_state:Code})))) % Verdict = ?_False
            ;   (initialize_monitor, term_string(TE1, TE1Str), Reply = _{}.put(E).put(_{verdict:false, monitor_state:'false_verdict', spec:TE1Str}))), % Verdict = False
            atom_json_dict(Json, Reply, [as(string)]),
            ws_send(WebSocket, string(Json)),
            manage_event(WebSocket),
            (   get_dict(terminate, E, true) ->  % Check if terminate is true after processing the event
            initialize_monitor,
            true  % This will end the recursion, effectively terminating the connection
        ;   manage_event(WebSocket)  % Continue processing the next event
        )
    ).


exception(undefined_global_variable, state, retry) :- trace_expression(_, TE), nb_setval(state, TE).
exception(undefined_global_variable, log, retry) :- (current_prolog_flag(argv, [_, LogFile|_]) -> open(LogFile, append, Stream); Stream = null), nb_setval(log, Stream).

:- current_prolog_flag(argv, [Spec|L]), L = [Port|_], asserta(port(Port)), use_module(Spec).
:- port(Port), writeln(Port), atom_number(Port, P), server(P).
