#!/usr/bin/env bash
# Source this file to configure the heracles_agents environment:
#   source setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Heracles agent settings
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Neo4j connection
export ADT4_HERACLES_IP="localhost"
export ADT4_HERACLES_PORT="7687"

echo "Environment ready  (venv=$(which python), HERACLES_AGENTS_PATH=${HERACLES_AGENTS_PATH})"
