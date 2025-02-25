# conda

极为特殊的环境管理工具

设计思想：conda将几乎所有的工具，第三方包都当作package来对待，甚至包括python和conda自身

它是一个虚拟化，从电脑独立开辟出来的环境。通俗的来讲，虚拟环境就是借助虚拟机docker来把一部分内容独立出来，而欧美把这部分独立出来的东西称作容器，在这个容器中，我们可以只安装我们需要的依赖包，各个容器之间相互隔离，互不影响

# 环境管理

## 查看虚拟环境列表

conda env list

列出当前系统中所有conda 创建的虚拟环境，即环境所在的目录。（好像是当前用户）

## 激活虚拟环境

conda activate env_name

## 退出虚拟环境

conda deactivate

不需要指定虚拟环境名

## 共享虚拟环境

 将虚拟环境中所有的依赖包统一导出一个配置文件中，在别的机器上使用这套代码的时候，根据conda导出的配置文件重建虚拟环境即可，这就是共享虚拟环境功能

conda env export --file python36_env.yml

environment.yaml是导出依赖的目标文件，运行命令后，当前目录下就回生成一个environment.yaml文件，包含了所有依赖信息。

根据配置文件创建虚拟环境

conda env create -f /home/chb/code/python36_env.yml

运行上述命令之后，在新的机器上也会创建出一个一摸一样的虚拟环境

## 删除虚拟环境

conda remove -n python36 --all

or

conda env remove -n python36

# 包管理

## 安装包

conda install package_name

或者

pip install package_name

## 列出所有的包

conda list

## 更新包

conda update package_name

如果想要一次性跟新所有的包

conda update --all

## 查找包

conda search keyword

例如：我们安装了pandas 但是忘记了准确的名字

conda search pan

## 删除包

conda remove package_name
