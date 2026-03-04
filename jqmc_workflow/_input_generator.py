"""TOML input file generator for jqmc.

Reads default parameters from ``jqmc.jqmc_miscs.cli_parameters`` and
merges user-supplied overrides before writing a TOML file.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the jqmc project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import copy
from logging import getLogger

import toml

from jqmc.jqmc_miscs import cli_parameters

logger = getLogger("jqmc-workflow").getChild(__name__)


def resolve_with_defaults(section_name: str, explicit_params: dict) -> dict:
    """Resolve *None* values using ``jqmc_miscs`` defaults and log them.

    Parameters
    ----------
    section_name : str
        TOML section name (``"vmc"``, ``"mcmc"``, ``"lrdmc-bra"``, ``"control"``).
    explicit_params : dict
        ``{param_name: value_or_None}``.  *None* entries are replaced
        by the corresponding default from ``cli_parameters``.

    Returns
    -------
    dict
        Resolved parameters with no *None* values (unless the default
        itself is *None*, which means the field is required).
    """
    defaults = cli_parameters.get(section_name, {})
    resolved = {}
    for key, value in explicit_params.items():
        if value is None:
            default_val = defaults.get(key)
            if default_val is not None:
                logger.info(f"  [{section_name}] {key} = {default_val} (default)")
            resolved[key] = default_val
        else:
            resolved[key] = value
    return resolved


def get_default_parameters(job_type: str) -> dict:
    """Return a deep-copied default parameter dict for *job_type*.

    The returned dict has two sections: ``"control"`` and the job-type
    section (e.g. ``"mcmc"``, ``"vmc"``, ``"lrdmc-bra"``, ``"lrdmc-tau"``).

    Parameters
    ----------
    job_type : str
        One of ``"mcmc"``, ``"vmc"``, ``"lrdmc-bra"``, ``"lrdmc-tau"``.

    Returns
    -------
    dict
        ``{"control": {...}, job_type: {...}}``
    """
    if job_type not in cli_parameters:
        raise ValueError(
            f"Unknown job_type '{job_type}'. Must be one of: {[k for k in cli_parameters if not k.endswith('_comments')]}"
        )
    params = {
        "control": copy.deepcopy(cli_parameters["control"]),
        job_type: copy.deepcopy(cli_parameters[job_type]),
    }
    params["control"]["job_type"] = job_type
    return params


def generate_input_toml(
    job_type: str,
    overrides: dict = None,
    filename: str = "input.toml",
    with_comments: bool = False,
) -> str:
    """Generate a jqmc TOML input file.

    Parameters
    ----------
    job_type : str
        One of ``"mcmc"``, ``"vmc"``, ``"lrdmc-bra"``, ``"lrdmc-tau"``.
    overrides : dict, optional
        Nested dict of values to override, e.g.
        ``{"control": {"number_of_walkers": 8}, "mcmc": {"num_mcmc_steps": 1000}}``.
    filename : str
        Output filename.
    with_comments : bool
        If True, insert inline comments from ``cli_parameters["*_comments"]``.

    Returns
    -------
    str
        The absolute path of the written file.

    Examples
    --------
    >>> generate_input_toml(
    ...     "mcmc",
    ...     overrides={"mcmc": {"num_mcmc_steps": 500, "Dt": 1.5}},
    ...     filename="mcmc.toml",
    ... )
    """
    params = get_default_parameters(job_type)
    overrides = overrides or {}

    for section, kvs in overrides.items():
        if section not in params:
            params[section] = {}
        if isinstance(kvs, dict):
            for k, v in kvs.items():
                params[section][k] = v
        else:
            # allow top-level scalar override (unusual but defensive)
            params[section] = kvs

    # Validate: required fields (those set to None) must be overridden
    for section, section_dict in params.items():
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                if v is None:
                    raise ValueError(
                        f"Required parameter '{k}' in [{section}] was not set. Please provide it via the 'overrides' dict."
                    )

    if with_comments:
        text = _dump_with_comments(params, job_type)
        with open(filename, "w") as f:
            f.write(text)
    else:
        with open(filename, "w") as f:
            toml.dump(params, f)

    logger.info(f"Generated {filename} (job_type={job_type})")
    return filename


def _dump_with_comments(params: dict, job_type: str) -> str:
    """Serialize *params* to TOML with inline comments."""
    lines = []
    for section, section_dict in params.items():
        lines.append(f"[{section}]")
        comment_key = f"{section}_comments"
        comments = cli_parameters.get(comment_key, {})

        if not isinstance(section_dict, dict):
            lines.append(f"{section} = {_toml_value(section_dict)}")
            lines.append("")
            continue

        for k, v in section_dict.items():
            comment = comments.get(k, "")
            if comment:
                # Truncate long comments
                if len(comment) > 120:
                    comment = comment[:117] + "..."
                lines.append(f"# {comment}")
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _toml_value(v) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, dict):
        # Inline table
        inner = ", ".join(f"{k} = {_toml_value(val)}" for k, val in v.items())
        return "{" + inner + "}"
    elif isinstance(v, list):
        inner = ", ".join(_toml_value(item) for item in v)
        return f"[{inner}]"
    else:
        return repr(v)
