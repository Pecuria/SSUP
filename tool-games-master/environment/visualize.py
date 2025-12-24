from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import demonstrateTPPlacement, visualizePath
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
    act_array = []
    policy_array = []
    prior = 0
    policy = 0
    time = len(act_array)
    
    while time < 100:
        time += 1
        dir = "Trials/Original"
        name = "/Catapult"
        task_dir = dir + name
        with open(task_dir + ".json", 'r') as f:
            btr = json.load(f)
        pgw = loadFromDict(btr["world"])
        tp = ToolPicker(btr)
        
        objects = pgw.getDynamicObjects()
        objects = verifyDynamicObjects(objects, btr)
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
        path_dict, success, _ = model.tp.observePlacementPath("obj2", [450., 450. ], model.maxtime)
        visualizePath(btr['world'],path_dict)
        exit(0)
        #print(path_dict, success)
        #model.simulate("obj3", (125.36628723144531, 336.5653991699219))
        #model.test()
        act_time, action_type = model.run()
        act_time -= 1
        act_array.append(act_time)
        if action_type == 'prior':
            prior += 1
        elif action_type == 'policy':
            policy += 1
            policy_array.append(act_time)
        print(act_array)
        print(policy_array)
        print(prior, policy)

        
    for _ in range(100):
        num[act_array[_]] += 1
    
    tot = 0
    for _ in range(21):
        tot += num[_]
        print(_, tot)
    #model.view_his()
        
    
if __name__ == '__main__':
    main()