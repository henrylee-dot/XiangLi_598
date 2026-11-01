#!/usr/bin/env python3
"""把录制目录里的 robot.bag 提取成 robot_extracted.npz，供 analyze_run.py 使用。

用法（系统 python3，需要先 source ~/ws_mini/devel/setup.bash）:
    python3 extract_bag.py <录制目录>
"""
import sys

import numpy as np
import rosbag

run = sys.argv[1]
js_t, js_q, ms_t, ms_p, ms_q, c_t, c_q = [], [], [], [], [], [], []
js_names = c_names = None
for topic, msg, t in rosbag.Bag(f'{run}/robot.bag').read_messages():
    if topic.endswith('joint_states'):
        js_names = js_names or list(msg.name)
        js_t.append(t.to_sec())
        js_q.append(msg.position)
    elif topic.endswith('model_states'):
        p = msg.pose[msg.name.index('robotis_mini')]
        ms_t.append(t.to_sec())
        ms_p.append([p.position.x, p.position.y, p.position.z])
        ms_q.append([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])
    else:
        c_names = c_names or list(msg.joint_names)
        c_t.append(t.to_sec())
        c_q.append(msg.points[0].positions)
np.savez(f'{run}/robot_extracted.npz', js_t=js_t, js_q=js_q, js_names=js_names,
         ms_t=ms_t, ms_p=ms_p, ms_q=ms_q, c_t=c_t, c_q=c_q, c_names=c_names)
print('提取完成:', len(js_t), '条关节状态,', len(c_t), '条命令')
