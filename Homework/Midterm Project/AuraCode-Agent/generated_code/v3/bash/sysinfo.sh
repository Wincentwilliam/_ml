#!/bin/bash
echo "=========================================="
echo "          SYSTEM INFORMATION              "
echo "=========================================="
echo -e "\n[OS Information]"
[ -f /etc/os-release ] && grep "PRETTY_NAME" /etc/os-release | cut -d'=' -f2 | tr -d '"' || uname -sr
echo -e "\n[User Information]"
echo "Current User: $(whoami)"
echo "Home Directory: $HOME"
echo "Shell: $SHELL"
echo -e "\n[Disk Usage]"
df -h / | awk 'NR==2 {print "Root Disk Usage: " $3 " used / " $2 " total (" $5 " used)"}'
echo "Detailed Partition Breakdown:"
df -h | grep -E '^/dev/' || echo "No /dev/ partitions found."
echo -e "\n=========================================="