#!/bin/sh

cd "$(dirname "$0")"

if [ -f ".env" ]; then
    source .env
fi

/bin/sleep 100000
