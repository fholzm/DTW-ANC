#!/usr/bin/env bash
set -euo pipefail

# Start JACK if not already running
if ! pgrep -x jackd >/dev/null 2>&1; then
  nohup jackd -m -d dummy -r 16000 -p 16 > /tmp/jackd.log 2>&1 &
  sleep 0.5
fi

# Keep the container alive
exec sleep infinity