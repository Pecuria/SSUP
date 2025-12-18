import numpy as np
import math
from scipy import stats
import torch
from torch import distributions as D
from collections import OrderedDict
torch.autograd.set_detect_anomaly(True)
from pyGameWorld.viewer import demonstrateTPPlacement
from pyGameWorld.helpers import distanceToObject

class SSUP:
    
    def __init__(
        self,
        objects,
        tp,
        tools,
        task_type,
        goal,
        goal_name,
        obj_in_goal_name,
        std_y = 200,
        std_x = 10,
        n_sims = 4,
        n_init = 5,
        epsilon = 0.3,
        dir_std = 0.1,
        ela_std = 0.1,
        lr = 0.3,
        scale_var = 1,
        T = 1.5,
        n_iter = 5
    ):
        self.objects = objects
        self.task_type = task_type
        self.goal_name = goal_name
        self.goal = goal
        self.tp = tp
        self.tools = OrderedDict(tools)
        self.oign = obj_in_goal_name
        self.std_y = std_y
        self.std_x = std_x
        self.n_sims = n_sims
        self.n_init = n_init
        self.epsilon = epsilon
        self.dir_std = dir_std
        self.ela_std = ela_std
        self.scale_var = scale_var
        self.ave_reward = 0
        self.lr = lr
        self.maxtime = 40.
        self.T = T
        self.n_iter = n_iter
        self.eps = 1e-5
        self.idx_to_name = {i : tool for i, tool in enumerate(self.tools.keys())}
        self.name_to_idx = {tool : i for i, tool in enumerate(self.tools.keys())}
        self.locs = torch.tile(torch.tensor([0.]),(3, 2)).requires_grad_(True)
        self.scale = torch.tile(torch.log(torch.tensor([scale_var])),(3,2)).requires_grad_(True)
        self.tool_logits = torch.ones(3).requires_grad_(True)
        self.optimizer = torch.optim.SGD([
            {'params': [self.tool_logits, self.locs], 'lr': self.lr},
            {'params': [self.scale], 'lr': self.lr * 0.2} # 降低 scale 的更新速度
        ])
        self.action_array = []
    
    def get_bounding_box(self, vertices):
        verts = np.array(vertices)
        #print(vertices)
        #print(verts)
        if verts.ndim == 2:
            verts = np.expand_dims(verts, axis=0)
        x_min, x_max = np.min(verts[:,:,0]), np.max(verts[:,:,0])
        y_min, y_max = np.min(verts[:,:,1]), np.max(verts[:,:,1])
        return x_min, y_min, x_max, y_max
    
    def oriented_prior(self, tool_name):
        tool_bb = self.get_bounding_box(self.tools[tool_name])
        tool_h = tool_bb[3] - tool_bb[1]
        obj_idx = np.random.choice(np.arange(len(self.objects)), p=np.ones(len(self.objects))/len(self.objects))
        obj = self.objects[obj_idx]
        obj_y = obj.getPos()[1]
        y_leftsigma = (tool_h / 2 - obj_y) / self.std_y
        y_rightsigma = (600 - tool_h / 2 - obj_y) / self.std_y
        if obj.type == 'Ball':
            x_left = obj.getPos()[0] - obj.radius
            x_right = obj.getPos()[0] + obj.radius
        else:
            x_left, _ , x_right, _ = self.get_bounding_box(obj.toGeom())
        
        #print(x_left, x_right)
        
        collide = True
        iter_y = 0
        a = -1
        while collide:
            if iter_y >= 5 or a == -1:
                a = np.random.rand()
                iter_y = 0
            iter_y += 1
            if a < 0.5 and y_rightsigma > 0:
                pos_y = stats.truncnorm.rvs(0, y_rightsigma, loc = obj_y, scale = self.std_y)
            else:
                pos_y = stats.truncnorm.rvs(y_leftsigma, 0, loc = obj_y, scale = self.std_y)
            v = stats.uniform.rvs(loc = x_left - self.std_x, scale = x_right - x_left + 2 * self.std_x)
            #print(v)
            if v < x_left:
                pos_x = stats.norm.rvs(loc = x_left, scale = self.std_x)
            elif v > x_right:
                pos_x = stats.norm.rvs(loc = x_right, scale = self.std_x)
            else:
                pos_x = v
            collide = self.tp.checkPlacementCollide(tool_name, [pos_x, pos_y])
        
        return [pos_x, pos_y]
    
    def initialize(self):
        #print(2)
        ave_init = 2
        for idx, tool_name in enumerate(self.tools.keys()):
            for _ in range(ave_init):
                pos = self.oriented_prior(tool_name)
                self.ave_reward += self.simulate(tool_name, pos)
        self.ave_reward /= ave_init * 3
        for idx, tool_name in enumerate(self.tools.keys()):
            #print(self.n_init)
            for _ in range(self.n_init):
                pos = self.oriented_prior(tool_name)
                reward = self.simulate(tool_name, pos)
                #print(tool_name, pos, reward, self.ave_reward, self.locs, self.tool_logits, self.scale, "initialize")
                #demonstrateTPPlacement(self.tp, tool_name, pos, hz = 150.)
                self.update(self.get_log(tool_name, pos), reward)
                
    
    def simulate(self, tool_name, pos):
        reward = 0
        if self.tp.checkPlacementCollide(tool_name, pos):
            return reward
        #print(tool_name, pos, self.dir_std, self.ela_std)
        for _ in range(self.n_sims):
            path_dict, success, _ = self.tp.runNoisyPath(
                toolname = tool_name,
                position = pos,
                maxtime = self.maxtime,
                noise_collision_direction = self.dir_std,
                noise_collision_elasticity = self.ela_std
            )
            if success:
                reward += 1
            else:
                reward += self.get_reward(path_dict)
        return 1.0 *  reward / self.n_sims
    
    def sample_prior(self):
        print("prior")
        toollist = list(self.tools.keys())
        tool = np.random.choice(np.arange(len(self.tools)), p=np.ones(len(self.tools))/len(self.tools))
        tool_name = toollist[tool]
        pos = self.oriented_prior(tool_name)
        return tool, pos
        
    def sample_policy(self):
        print("policy")
        tool_dist = D.Categorical(logits = self.tool_logits)
        tool = tool_dist.sample()
        collide = True
        while collide:
            pos_dist_x = D.Normal(loc = self.locs[tool][0], scale = torch.exp(self.scale[tool][0]))
            pos_dist_y = D.Normal(loc = self.locs[tool][1], scale = torch.exp(self.scale[tool][1]))
            pos = torch.tensor([pos_dist_x.sample()*75+300,pos_dist_y.sample()*75+300])
            cur_pos = pos.detach().numpy()
            collide = self.tp.checkPlacementCollide(toolname=list(self.tools.keys())[tool.item()], position=(float(cur_pos[0]), float(cur_pos[1])))
        return tool.item(), (float(cur_pos[0]),float(cur_pos[1]))
    
    def sample_action(self):
        a = np.random.rand()
        #print (a, self.epsilon, "action")
        if a < self.epsilon:
            return self.sample_prior()
        else:
            return self.sample_policy()
        
    def get_reward(self, path_dict):
        reward = 0
        for obj in self.oign:
            tra = np.array(path_dict[obj])
            dis_init = distanceToObject(self.goal, tra[0])
            min_dis = dis_init
            for point in tra:
                dis = distanceToObject(self.goal, point)
                if dis < min_dis:
                    min_dis = dis
            frac = min_dis / dis_init
            if 1 - frac > reward:
                reward = 1 - frac
        #return reward
        return reward * reward
    
    def get_log(self, toolname, pos):
        tool_idx = self.name_to_idx[toolname]
        pos = [(pos[0] - 300)/75, (pos[1] - 300)/75]
        tool_dist = D.Categorical(logits = self.tool_logits)
        pos_dist_x = D.Normal(loc = self.locs[tool_idx][0], scale = torch.exp(self.scale[tool_idx][0]))
        pos_dist_y = D.Normal(loc = self.locs[tool_idx][1], scale = torch.exp(self.scale[tool_idx][1]))
        log_prob_tool = tool_dist.log_prob(torch.tensor(tool_idx))
        log_prob_pos_x = pos_dist_x.log_prob(torch.tensor(pos[0]))
        log_prob_pos_y = pos_dist_y.log_prob(torch.tensor(pos[1]))
        #print(log_prob_tool, log_prob_pos, "get_log")
        return (log_prob_tool, log_prob_pos_x + log_prob_pos_y)
    
    def update(self, log_probs, reward):
        self.optimizer.zero_grad()
        loss = -((reward - self.ave_reward) / (1 - self.ave_reward)) * (log_probs[1] + log_probs[0])
        self.ave_reward = 0.9 * self.ave_reward + 0.1 * reward
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for i in range(3):
                for t in range(2):
                    if -1 > self.scale[i][t]:
                        self.scale[i][t] = -1
                    if 1 < self.scale[i][t]:
                        self.scale[i][t] = 1
                    if -3.5 > self.locs[i][t]:
                        self.locs[i][t] = -3.5
                    if 3.5 < self.locs[i][t]:
                        self.locs[i][t] = 3.5
            
    
    def view_his(self):
        for tool, pos in self.action_array:
            print(tool, pos)
            demonstrateTPPlacement(self.tp, tool, pos, hz = 150.)
    
    def test(self):
        succ = 0
        for i in range(100):
            tool_idx, pos = self.sample_prior()
            tool_name = list(self.tools.keys())[tool_idx]
            path_dict , success, _ = self.tp.observePlacementPath(tool_name, pos, self.maxtime)
            demonstrateTPPlacement(self.tp, tool_name, pos, maxtime = 20., hz = 150.)
            if success:
                succ += 1
            
        print(succ)
    
    def run(self):
        print(1)
        self.initialize()
        success = False
        best_reward = -1.0
        best_pos = None
        best_toolname = None
        act_time = 0
        it = 0
        while not success:
            acting = False
            tool_idx, pos = self.sample_action()
            tool_name = list(self.tools.keys())[tool_idx]
            reward = self.simulate(tool_name, pos)
            print(tool_idx, tool_name, pos, reward, self.ave_reward, self.locs, self.scale, self.tool_logits, "sample")
            #demonstrateTPPlacement(self.tp, tool_name, pos, maxtime = self.maxtime, hz = 90.)
            if reward > best_reward:
                #print("update")
                best_reward = reward
                best_toolname = tool_name
                best_pos = pos
            it += 1
            if reward > self.T:
                acting = True
                #print(tool_name, pos, reward, "better")
                path_dict , success, _ = self.tp.observePlacementPath(tool_name, pos, self.maxtime)
                self.action_array.append([tool_name, pos])
            elif it >= self.n_iter:
                acting = True
                #print(best_toolname, best_pos, "best")
                tool_name, pos = best_toolname, best_pos
                path_dict, success, _ = self.tp.observePlacementPath(tool_name, pos, self.maxtime)
                self.action_array.append([best_toolname,best_pos])
            
            if acting:
                real_reward = self.get_reward(path_dict)
                act_time += 1
                if real_reward > self.T:
                    self.T = real_reward
                print(tool_idx, tool_name, pos, real_reward, self.ave_reward, self.locs, self.scale, self.tool_logits, "toolname")
                #demonstrateTPPlacement(self.tp, tool_name, pos, maxtime = self.maxtime, hz = 90.)
                if success:
                    print(act_time, "action was successful")
                    return act_time
                if act_time > 20:
                    print("fail")
                    return act_time
                self.update(self.get_log(tool_name, pos), real_reward)
                
                for t in self.tools.keys():
                    if tool_name != t:
                        another_reward = self.simulate(t, pos)
                        self.update(self.get_log(t, pos), another_reward)
                
                it = 0
                best_reward = -1.0
                best_toolname = None
                best_pos = None
            
            else:
                self.update(self.get_log(tool_name, pos), reward)
                
                