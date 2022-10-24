with import <nixpkgs> {};
let ps = python3Packages;
in pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
    gcc

    # A Python interpreter including the 'venv' module is required to bootstrap
    # the environment.
    ps.python

    # This execute some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    ps.venvShellHook

    # Those are dependencies that we would like to use from nixpkgs, which will
    # add them to PYTHONPATH and thus make them accessible from within the venv.
    ps.pylev
    ps.appdirs
    ps.pyparsing
    ps.pastel
    ps.numpy
    ps.multidict
    ps.idna
    ps.frozenlist
    ps.crashtest
    ps.charset-normalizer
    ps.attrs
    ps.async-timeout
    ps.yarl
    ps.packaging
    ps.clikit
    ps.aiosignal
    ps.marshmallow
    ps.aiohttp
    ps.webargs
    ps.pandas
    ps.arviz
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${lib.makeLibraryPath [stdenv.cc.cc.lib]}"
  '';
}
