# How to write jQMC documentation

This directory contains python-sphinx documentation source.

## How to compile

```
make html
```

## Source files

* `conf.py` contains the sphinx setting confiuration.
* `*.rst` are restructuredtext documentation source files.
* `*.md` are markdown documentation source files.

## How to publish

Web page files are copied to `gh-pages` branch. At the jQMC github top directory,
```
git checkout gh-pages
rm -r .buildinfo .doctrees *
```

In the directory, please compile the sphinx doc.,
```
rsync -avh _build/ <jqmc-repository-directory>/
```

Again, at the jqmc github top directory,
```
git add .
git commit -a -m "Update documentation ..."
git push
```
