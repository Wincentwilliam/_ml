#!/bin/bash

echo "=========================================="
echo "          SYSTEM INFORMATION             "
echo "=========================================="

# Get OS Name
OS_NAME=$(uname -s)
echo "OS Name:         $OS_NAME"

# Get Current User
CURRENT_USER=$(whoami)
echo "Current User:    $CURRENT_USER"

# Get Home Directory
HOME_DIR=$HOME
echo "Home Directory:   $HOME_DIR"

echo "------------------------------------------"
echo "Disk Usage:"
# Display disk usage for the root filesystem in human-readable format
df -h / | awk 'NR==2 {print "Root Partition Usage: " $5 " used (" $4 " available)"}'

echo "=========================================="