#!/bin/sh
# Permission-ready transport. Do not run until the user approves the four-file upload.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET=/home/bsm/tmp/hexset-port-liquidity-15e00c6b295792c1
FILES='src/hexset/bots/heximax/port_liquidity.py src/hexset/catanatron/port_liquiditycfg.py src/hexset/catanatron/port_liquidity_runner.py scripts/port_liquidity_queue.sh'
SSH='ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 wintermute'
$SSH "wsl.exe -d Debian -- sh -lc 'set -eu; test -d /home/bsm/tmp/hexset-port-liquidity-15e00c6b295792c1/src; mkdir -p $TARGET/src/hexset/bots/heximax $TARGET/src/hexset/catanatron'"
tar -C "$ROOT" -cf - $FILES | $SSH "wsl.exe -d Debian -- tar -x -C $TARGET"
$SSH "wsl.exe -d Debian -- sh -lc 'set -eu; cd $TARGET; printf \"%s  %s\\n\" e9ab7a80922a3230795fa8351e566baa93a215d6c65f0cd3dff66590233af358 src/hexset/bots/heximax/port_liquidity.py 686894bdfe601985ac6d9935f8fe963c58fb515ba8e2bcb4846b9064a685fcf5 src/hexset/catanatron/port_liquiditycfg.py da8e757edc01582068eb2b6ac57ae5a4b2001918cc3b8ed6f5e28e2e2a729c98 src/hexset/catanatron/port_liquidity_runner.py 01dd6dac19ee20806a187edcae9ce33f36448a7a43cb53d5941b70238277c538 scripts/port_liquidity_queue.sh | sha256sum -c -'"
