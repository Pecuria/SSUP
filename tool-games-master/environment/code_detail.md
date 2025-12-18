code_detail

世界文件结构：

```json
"world":
	"dims": 世界大小 [600,600]
	"bts"：不知道 0.01
	"gravity"：重力 200
	"defaults":
		"density": 1
		"friction": 0.5
		"elasticity": 0.5
		"color": black
		"bk_color": white
	"objects":
		放各种各样的 object
	"blocks":
		放block
	"constraints"：
		放constraints
	"gcond":
		"type"
			包含 specificInGoal（一个obj，默认为 "Ball"） 
			或 ManyInGoal（很多 obj，用 array 存）
		"goal"
			包含作为 goal 的 obj 的名字（key）
		"obj"
			需要进入 goal 的 obj 的名字（key）
		"duration"
"tools":
	"obj1":
		...
	"obj2":
		...
	"obj3":
		...
	"toolNames":
		放toolname的	["obj1","obj2","obj3"]
	"sucText":
		任务描述 "Get the red ball into the green goal"
```



 一些能用到的函数：

```python
tp.observePlacementPath(toolname, position, maxtime)
#把 toolname 放在 position 上模拟
#返回 path_dict, success, time_to_success
#分别表示路径点，是否成功，成功时间
#tp 是 toolpicker 的缩写
```

```python
object.getPos() #返回一个 object 的位置，return np.array[(x,y)]
```

object 包含七种 type: `Poly, Ball, Container, Compound, Goal, Blocker, Segment`

可以使用 toGeom 函数获得 Poly, Container, Compound  的顶点，获得 Ball 的位置和半径

````python
tp.checkPlacementCollide(toolname, position = [x,y])
#返回 toolname 在 position 是否和场景碰撞，若是返回 True
````

```python
tp.runNoisyPath(toolname, position, maxtime
	noise_collision_direction,#direction sample 方差
	noise_collision_elasticity)#同上
#和 tp.observePlacementPath 返回差不多
```

```python
from pyGameWorld import PGWorld, ToolPicker
with open(json_dir+tnm+'.json','r') as f:
  btr = json.load(f)
obj = loadFromDict(btr["world"]) #PGWorld
tp = ToolPicker(btr) #ToolPicker
```

```python
pgw.getDynamicObjects() #返回所有可移动的 obj 的名字组成的 array
```

```python
#helpers.py 有很多有用的函数
distanceToObject(object, point)
#返回 point 到 object 的距离，不需要自己实现
```

