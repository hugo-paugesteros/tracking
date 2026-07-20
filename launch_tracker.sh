#!/bin/bash
# Double-click this file (or right-click > Run) to launch the tracker.
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "No 'venv' folder found here."
    echo "Please follow the setup steps in README.md first (creating the"
    echo "virtual environment and running 'pip install -e .')."
    echo ""
    read -p "Press Enter to close this window..."
    exit 1
fi

source venv/bin/activate
launch-tracker

echo ""
read -p "Press Enter to close this window..."
