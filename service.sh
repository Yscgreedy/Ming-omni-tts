#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/runtime-cache/service-pids"

NUM_WORKERS="${NUM_WORKERS:-1}"
BASE_PORT="${BASE_PORT:-8000}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-start}"

mkdir -p "${PID_DIR}"

pid_file_for_port() {
  local port="$1"
  echo "${PID_DIR}/service-${port}.pid"
}

is_pid_running() {
  local pid="$1"
  kill -0 "${pid}" >/dev/null 2>&1
}

cleanup_stale_pid_files() {
  for pid_file in "${PID_DIR}"/service-*.pid; do
    [ -e "${pid_file}" ] || continue
    local pid
    pid="$(cat "${pid_file}")"
    if ! is_pid_running "${pid}"; then
      rm -f "${pid_file}"
    fi
  done
}

start_services() {
  cleanup_stale_pid_files
  echo "${CNB_VSCODE_PROXY_URI:-}"
  for i in $(seq 1 "${NUM_WORKERS}"); do
    local port pid_file pid
    port=$((BASE_PORT + i))
    pid_file="$(pid_file_for_port "${port}")"
    if [ -f "${pid_file}" ]; then
      pid="$(cat "${pid_file}")"
      if is_pid_running "${pid}"; then
        echo "service on port ${port} already running (pid=${pid})"
        continue
      fi
      rm -f "${pid_file}"
    fi

    PORT="${port}" "${PYTHON_BIN}" "${SCRIPT_DIR}/service/app.py" &
    pid=$!
    echo "${pid}" > "${pid_file}"
    echo "started service on port ${port} (pid=${pid})"
    sleep "${STARTUP_WAIT_SECONDS}"
  done

  echo "ALL ${NUM_WORKERS} started"
}

stop_services() {
  cleanup_stale_pid_files
  local stopped_any=0
  for pid_file in "${PID_DIR}"/service-*.pid; do
    [ -e "${pid_file}" ] || continue
    local pid port
    pid="$(cat "${pid_file}")"
    port="$(basename "${pid_file}" .pid | sed 's/service-//')"
    if is_pid_running "${pid}"; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
      echo "stopped service on port ${port} (pid=${pid})"
      stopped_any=1
    fi
    rm -f "${pid_file}"
  done

  if [ "${stopped_any}" -eq 0 ]; then
    echo "no managed service process is running"
  fi
}

status_services() {
  cleanup_stale_pid_files
  local running_count=0
  for pid_file in "${PID_DIR}"/service-*.pid; do
    [ -e "${pid_file}" ] || continue
    local pid port
    pid="$(cat "${pid_file}")"
    port="$(basename "${pid_file}" .pid | sed 's/service-//')"
    if is_pid_running "${pid}"; then
      echo "running: port=${port} pid=${pid}"
      running_count=$((running_count + 1))
    fi
  done

  if [ "${running_count}" -eq 0 ]; then
    echo "no managed service process is running"
  fi
}

case "${ACTION}" in
  start)
    start_services
    ;;
  stop|shutdown)
    stop_services
    ;;
  restart)
    stop_services
    start_services
    ;;
  status)
    status_services
    ;;
  *)
    echo "usage: $0 {start|stop|shutdown|restart|status}" >&2
    exit 1
    ;;
esac
