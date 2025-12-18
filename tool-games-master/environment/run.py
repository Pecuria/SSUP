from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import demonstrateTPPlacement
import json
import pygame as pg
import numpy as np
import os
from ssup import SSUP

def verifyDynamicObjects(objects, trial):
    if not trial['world']['blocks']:
        return objects
    else:
        blocker_verts = trial['world']['blocks']['Blocker']['vertices']
        
    new_objs = []
    for obj in objects:
        obj_pos = obj.getPos()
        if obj_pos[0] < blocker_verts[0][0] or obj_pos[0] > blocker_verts[2][0] or obj_pos[1] < blocker_verts[0][1] or obj_pos[1] > blocker_verts[2][1]:
            new_objs.append(obj)
    return new_objs

def main():
    num = np.zeros(21)
    act_array = [20, 1, 1, 10, 2, 20, 4, 13, 0, 20, 20, 4, 20, 16, 10, 13, 11, 20, 15, 20, 0, 12, 20, 0, 5, 11, 8, 16, 6, 20, 16, 4, 20, 8, 12, 10, 20, 11, 20, 16, 1, 2, 20, 8, 2, 20, 3, 6, 20, 14, 1, 7, 18, 1, 18, 2, 20, 20, 5, 2, 7, 20, 20, 6, 5, 11, 0, 20, 10, 19, 11, 14, 3, 0, 12, 2, 7, 20, 3, 10, 5, 20, 6, 10, 3, 0, 7, 12, 20, 20, 20]
    time = len(act_array)
    
    while time < 100:
        time += 1
        dir = "Trials/Original"
        name = "/Bridge"
        task_dir = dir + name
        with open(task_dir + ".json", 'r') as f:
            btr = json.load(f)
        pgw = loadFromDict(btr["world"])
        tp = ToolPicker(btr)
        
        objects = pgw.getDynamicObjects()
        tools = tp._tools
        goal_name = btr["world"]["gcond"]["goal"]
        goal = pgw.objects[goal_name]
        task_type = btr["world"]["gcond"]["type"]
        if task_type == 'SpecificInGoal':
            obj_in_goal_name = [btr["world"]["gcond"]["obj"]]
        elif task_type == 'ManyInGoal':
            obj_in_goal_name = btr["world"]["gcond"]["objlist"]
        print(time,"%")
        model = SSUP(objects = objects, tools = tools, goal = goal, task_type = task_type, goal_name = goal_name, tp = tp, obj_in_goal_name = obj_in_goal_name, epsilon = 0.3)
        #model.test()
        act_time = model.run() - 1
        act_array.append(act_time)
        print(act_array)
        
    for _ in range(100):
        num[act_array[_]] += 1
    
    tot = 0
    for _ in range(21):
        tot += num[_]
        print(_, tot)
    #model.view_his()
        
    
if __name__ == '__main__':
    main()