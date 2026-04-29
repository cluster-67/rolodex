#!/bin/bash


ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
SCRIPTS_DIR=$ROOT_DIR/scripts
BUILD_DIR=$ROOT_DIR/build

color_echo () {
  local color="$1"
  shift

  case "$color" in
    red)    code="31" ;;
    green)  code="32" ;;
    yellow) code="33" ;;
    blue)   code="34" ;;
    *)      code="0" ;;
  esac

  echo -e "\033[${code}m$*\033[0m"
}
