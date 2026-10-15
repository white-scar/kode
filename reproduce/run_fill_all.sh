#!/bin/sh
nohup sh /data/fandata/dashuaige/reproduce/_work_run0.sh > /dev/null 2>&1 & echo "GPU0 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run1.sh > /dev/null 2>&1 & echo "GPU1 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run2.sh > /dev/null 2>&1 & echo "GPU2 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run3.sh > /dev/null 2>&1 & echo "GPU3 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run4.sh > /dev/null 2>&1 & echo "GPU4 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run5.sh > /dev/null 2>&1 & echo "GPU5 PID=$!"
nohup sh /data/fandata/dashuaige/reproduce/_work_run6.sh > /dev/null 2>&1 & echo "GPU6 PID=$!"
