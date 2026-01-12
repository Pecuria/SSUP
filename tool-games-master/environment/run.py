from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import demonstrateTPPlacement, demonstrateWorld
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
    if len(new_objs) == 0:
        return objects
    return new_objs

def set_model(dir, name, up_down, truncnorm):
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
    model = SSUP(objects = objects, tools = tools, goal = goal, task_type = task_type, goal_name = goal_name, tp = tp, obj_in_goal_name = obj_in_goal_name, epsilon = 0.3, up_down = up_down, truncnorm = truncnorm)
    return model, pgw, tp

def prior_test():
    names = ["/Table_B"]
    bits = [True, False]
    
    for name in names:
        for up_down in bits:
            for truncnorm in  bits:
                model , _ , _ = set_model(dir = "Trials/Original", name = name, up_down = up_down, truncnorm = truncnorm)
                succ = model.test()
                print(name, up_down, truncnorm, succ)
    
    exit(0)
    
def main():
    num = np.zeros(21)
    act_array = []
    policy_array = []
    prior = 0
    policy = 0
    first_action = []
    last_action = []
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
        demonstrateWorld(pgw, hz = 9999999999., action = last_action,draw=True)
        exit(0)
        
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
        
        act_time, action_type = model.run()
        first_action.append(model.first)
        last_action.append(model.last)
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
        print(first_action)
        print()
        print(last_action)
        if time == 100:
            demonstrateWorld(pgw, hz = 9999999999., action = first_action, draw = True)
            demonstrateWorld(pgw, hz = 9999999999., action = last_action, draw = True)
            exit(0)

        
    for _ in range(len(act_array)):
        num[act_array[_]] += 1
    
    tot = 0
    for _ in range(21):
        tot += num[_]
        print(_, tot)
    #model.view_his()
        
    
if __name__ == '__main__':
    main()