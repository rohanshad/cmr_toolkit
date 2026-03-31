#!/bin/bash
# Run once after cloning: bash hooks/install.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

ln -sf "$REPO_ROOT/hooks/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"

echo "Installed: pre-push"
echo ""
echo "Note: tests/run_docker_tests.sh is workstation-guarded and will only"
echo "run the full Docker suite on jarvis_lambda. On other machines the"
echo "pre-push hook will exit with an error — disable or skip if needed."
