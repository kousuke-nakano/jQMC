# jQMC
jQMC code implements two real-space ab initio quantum Monte Carlo (QMC) methods.
Variatioinal Monte Carlo (VMC) and lattice regularized diffusion Monte Carlo (LRDMC) methods.

## Developers
Kosuke Nakano (National Institute for Materials Science [NIMS], Japan)

## Sponsors
<img src="https://www.jst.go.jp/kisoken/presto/wp-content/themes/02_presto/img/logo-ja.png" />


## How to install jQMC

jQMC can be installed via pip

```bash
% pip install .
```

## Documentation

jQMC user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.

## Develpment branch

The development of jQMC is managed on the `develop` branch of github jQMC repository.

- Github issues is the place to discuss about jQMC issues.
- Github pull request is the place to request merging source code.

## Formatting

Formatting rules are written in `pyproject.toml`.

## pre-commit

Pre-commit (https://pre-commit.com/) is mainly used for applying the formatting
rules automatically. Therefore, it is strongly encouraged to use it at or before
git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see
  https://pre-commit.com/#install.
- pre-commit hook is installed by `pre-commit install`.
- pre-commit hook is run by `pre-commit run --all-files`.

Unless running pre-commit, pre-commit.ci may push the fix at PR by github
action. In this case, the fix should be merged by the contributor's repository.

## VSCode setting
- Not strictly, but VSCode's `settings.json` may be written like below

  ```json
  "ruff.lint.args": [
      "--config=${workspaceFolder}/pyproject.toml",
  ],
  "[python]": {
      "editor.defaultFormatter": "charliermarsh.ruff",
      "editor.codeActionsOnSave": {
          "source.organizeImports": "explicit"
      }
  },
  ```

## How to run tests

Tests are written using pytest. To run tests, pytest has to be installed.
The tests can be run by

```bash
% pytest -v
```
