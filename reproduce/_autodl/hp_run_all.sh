#!/bin/bash
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work0.sh > /dev/null 2>&1 < /dev/null & echo "WORK0 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work1.sh > /dev/null 2>&1 < /dev/null & echo "WORK1 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work2.sh > /dev/null 2>&1 < /dev/null & echo "WORK2 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work3.sh > /dev/null 2>&1 < /dev/null & echo "WORK3 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work4.sh > /dev/null 2>&1 < /dev/null & echo "WORK4 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work5.sh > /dev/null 2>&1 < /dev/null & echo "WORK5 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work6.sh > /dev/null 2>&1 < /dev/null & echo "WORK6 PID=$!"
setsid nohup bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_work7.sh > /dev/null 2>&1 < /dev/null & echo "WORK7 PID=$!"
sleep 3; pgrep -af knode_hyperparam_ablation | head -12
