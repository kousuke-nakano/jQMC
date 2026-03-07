# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
APIDOC_DIR   = ./api_reference_mod
MODULES_RST  = $(APIDOC_DIR)/modules.rst
APIDOC_TITLE = API reference for modules

apidoc:
	sphinx-apidoc -f -o ./$(APIDOC_DIR)/ ../jqmc/
	sphinx-apidoc -f -T -o ./$(APIDOC_DIR)/ ../jqmc_workflow/
	@python3 -c 'import sys,pathlib,re;NL=chr(10);f=pathlib.Path(sys.argv[1]);title=sys.argv[2];lines=f.read_text("utf-8").splitlines(True);allowed="=~-^#*+";i=next(i for i in range(len(lines)-1) if (t:=lines[i].strip()) and (u:=lines[i+1].strip()) and len(set(u))==1 and all(c in allowed for c in u));ch=u[0];lines[i]=title+NL;lines[i+1]=ch*max(len(title),80)+NL;text="".join(lines);text=re.sub(r"(\.\. _api_ref_workflow:\s*)+","",text);text=".. _api_ref_workflow:"+NL+NL+text;text=text if "jqmc_workflow" in text else text.rstrip()+NL+"   jqmc_workflow"+NL;f.write_text(text,"utf-8")' "$(MODULES_RST)" "$(APIDOC_TITLE)"

html: apidoc
	$(SPHINXBUILD) -b html . ../docs_build

%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
