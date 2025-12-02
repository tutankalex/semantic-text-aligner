{ pkgs ? import <nixpkgs> {} }:
let
  nix_shortcuts = import (pkgs.fetchurl {
    url = "https://raw.githubusercontent.com/whacked/setup/refs/heads/master/bash/nix_shortcuts.nix.sh";
    hash = "sha256-jLbvJ52h12eug/5Odo04kvHqwOQRzpB9X3bUEB/vzxc=";
  }) { inherit pkgs; };

in pkgs.mkShell {
  buildInputs = [
    pkgs.rlwrap
    pkgs.python3
    pkgs.uv
  ] ++ [
    pkgs.nodejs
    pkgs.readline
  ]
  ++ nix_shortcuts.buildInputs
  ;  # join lists with ++

  shellHook = nix_shortcuts.shellHook + ''
    LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
    ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

    WORKDIR=$PWD
    if [ ! -e "$WORKDIR/.venv" ]; then
      uv venv .venv
      uv init
    fi

    source .venv/bin/activate
    export PATH=$PATH:"$WORKDIR/.venv/bin"
    export PATH=$PATH:/opt/npm/bin
    export DO_NOT_TRACK=1
  '' + ''
    echo-shortcuts ${__curPos.file}
    unset shellHook
  '';  # join strings with +
}
