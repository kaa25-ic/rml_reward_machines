#!/usr/bin/env bash

# MIT License
#
# Copyright (c) [2019] [Davide Ancona, Luca Franceschini, Angelo Ferrando, Viviana Mascardi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# get dir of this script
# https://stackoverflow.com/a/246128/1202636
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
swipl_bin="${SWIPL:-swipl}"

if ! command -v "$swipl_bin" >/dev/null 2>&1; then
    bundled_swipl="$here/../SWI-Prolog.app/Contents/MacOS/swipl"
    if [ -x "$bundled_swipl" ]; then
        swipl_bin="$bundled_swipl"
    fi
fi

if ! command -v "$swipl_bin" >/dev/null 2>&1; then
    project_bundled_swipl="$here/../../../legacy/SWI-Prolog.app/Contents/MacOS/swipl"
    if [ -x "$project_bundled_swipl" ]; then
        swipl_bin="$project_bundled_swipl"
    fi
fi

if ! command -v "$swipl_bin" >/dev/null 2>&1; then
    repo_bundled_swipl="$here/../../../SWI-Prolog.app/Contents/MacOS/swipl"
    if [ -x "$repo_bundled_swipl" ]; then
        swipl_bin="$repo_bundled_swipl"
    fi
fi

# specify monitor alias path for modules imported by the spec (like deep_subdict)
exec "$swipl_bin" -p monitor="$here" "$here"/online_monitor_edit_fast.pl -- "$@"
