# Source from Docker lifecycle scripts. Locates python / xdit; does not invent
# GPU-stack env (allocator, HIP, GPU_ARCHS, ROCM_PATH). Those belong to the image.

_xdit_find_python() {
  local candidate bindir xdit_bin

  if [ -n "${XDIT_PYTHON:-}" ] && [ -x "${XDIT_PYTHON}" ]; then
    echo "${XDIT_PYTHON}"
    return 0
  fi

  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return 0
  fi

  if [ -x /opt/venv/bin/python ]; then
    echo /opt/venv/bin/python
    return 0
  fi

  xdit_bin="$(command -v xdit 2>/dev/null || true)"
  if [ -n "${xdit_bin}" ]; then
    bindir="$(dirname "$(readlink -f "${xdit_bin}")")"
    if [ -x "${bindir}/python" ]; then
      echo "${bindir}/python"
      return 0
    fi
    if [ -x "${bindir}/python3" ]; then
      echo "${bindir}/python3"
      return 0
    fi
  fi

  for candidate in python3 python; do
    if candidate="$(command -v "${candidate}" 2>/dev/null)" && [ -x "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "xdit-env: could not find python (set XDIT_PYTHON or VIRTUAL_ENV)" >&2
  return 1
}

if ! PY="$(_xdit_find_python)"; then
  exit 1
fi
bindir="$(dirname "$(readlink -f "${PY}")")"
VENV="$(cd "${bindir}/.." && pwd)"

export PY
export VENV
export VIRTUAL_ENV="${VENV}"
export PATH="${bindir}:${PATH}"

if [ -x "${bindir}/xdit" ]; then
  export XDIT_BIN="${bindir}/xdit"
else
  export XDIT_BIN="$(command -v xdit 2>/dev/null || true)"
fi

_pip_constraint="${PIP_CONSTRAINT:-/opt/build-constraints.txt}"
if [ -f "${_pip_constraint}" ]; then
  export PIP_CONSTRAINT="${_pip_constraint}"
else
  unset PIP_CONSTRAINT
fi
unset _pip_constraint _xdit_find_python bindir xdit_bin candidate
