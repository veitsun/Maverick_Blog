### file(GLOB HEllOLIB_SRC_cpp ${PROJECT_SOURCE_DIR}/buffer_file_homework/src/*.cpp)

file(GLOB {name} ...) 命令是用来收集源文件的，并将他们存放在变量name中

- file是cmake中的一个命令

- GLOB是它的一个参数，表示全局匹配 ，GLOB用于查找文件，并将文件列表存储在一个变量中

- $ {PROJECT_SOURCE_DIR} 这是cmake自带的一个变量，表示项目根目录

### set命令

主要是用来定义和设置变量

``set(HELLOLIB_SRC $(HEllOLIB_SRC_cpp))``

- set 用来定义和修改变量的值

- HELLOLIB_SRC 是创建和修改的命名，HELLOLIB_SRC是变量

### add_library 命名

`add_library(SUNWEI SHARED ${HELLOLIB_SRC})`

- 是用来定义一个库（可以是共享库或者是静态库）。

- SUNWEI 是被定义库的名称。这个名称可以用于之后的链接和引用

- SHARED 这个是参数，作用是用来表示创建的是一个动态库，共享库在运行时可以被多个程序共享加载（相反STATIC）

- ${HELLOLIB_SRC}是一个变量，包含了源文件列表
