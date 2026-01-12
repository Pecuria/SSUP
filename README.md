README

环境配置：

* numpy
* python = 3.8
* scipy
* pymunk (=5.7) 
* execjs (installed by `pip install PyExecJS`)

You will also need node.js

The following package is not required, but you probably want it for visualization:

* pygame

Then just run the following code in this directory:

> python setup.py build

源代码存放在 ` ./tool-games-master/environment` 目录下， 主要结构为：

```
SSUP/tool-games-master/environment
├── Trials/            # 存放关卡数据 (你提到修改过这里)
├── pyGameWorld/	  # environment
├── ssup.py		       # ssup 算法部分
├── run.py		       #  主函数部分
└── draw.py			  # 绘制数据
```

可以直接使用 `./run.py` 运行