#!/bin/bash
# Script để check ai đang dùng GPU

echo "🔍 GPU Process Analysis:"
echo "========================"

# Lấy PID từ nvidia-smi và check owner
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | while read line
do
    pid=$(echo $line | cut -d',' -f1 | tr -d ' ')
    process=$(echo $line | cut -d',' -f2 | tr -d ' ')
    memory=$(echo $line | cut -d',' -f3 | tr -d ' ')
    
    if [ -n "$pid" ]; then
        # Get user info
        user=$(ps -o user= -p $pid 2>/dev/null)
        cmd=$(ps -o cmd= -p $pid 2>/dev/null)
        
        echo "📊 PID: $pid | User: ${user:-'Unknown'} | Memory: ${memory}MB"
        echo "   Process: $process"
        echo "   Command: ${cmd:-'N/A'}"
        echo "   ---"
    fi
done