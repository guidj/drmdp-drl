#!/bin/bash
set -e

uv sync
git add -u uv.lock
