#!/bin/bash
# Device-level GPU utilisation on Apple silicon, no root needed (IOAccelerator PerformanceStatistics).
ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'
