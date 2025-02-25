# Proto3语法

## 定义消息类型

### 定义一个搜索请求的消息格式

> 检索字段
> 
> 特定的结果页（感兴趣的结果所在的页面）
> 
> 每个页面的结果数量

```
syntax = "proto3";


message SearchRequest {
    string query = 1; // 检索字段
    int32 page_number = 2; // 特定的结果页
    int32 result_per_page = 3; // 每个页面的结果数量
}
```

`syntax = "proto3";` 必须要写在非空，非注释的第一行

`SearchRequest`  消息明确定义了三个字段（键值对），对应每一条你想包含在这个消息类型中的数据。每个字段都有一个名称和类型。

### 指明字段类型

在上面的例子中，所有的字段都是明确类型的：两个integers（page_number 和 result_per_page）和一个string (query)

## 分配字段序号

在定义的消息中每个字段都有一个唯一的序号。

这些序号用来在二进制消息结果中标识你的字段，而且一旦使用了消息类型，就不应该再变动。

字段序号在1到15的范围内占用一个字节来编码，包括字段序号和字段类型

字段序号在16到2047范围内占两个字节。

所以应该为经常使用的消息元素保留1到15的序号。切记为将来可能新增的常用元素预留了一些空间

### 标明字段规则

消息字段可以遵循下列规则之一

singular：符合语法规则的消息可以拥有0个或者1个该字段

repeated：该字段可以重复任意次数（包括0次）。重复变量的顺序将被保留。

### 新增更过消息类型

### 保留字段

### 你的`.proto`文件会发生什么

当使用protocol buffer 编译器时候，编译器会根据你选定的语言来生成你`.proto`文件中描述的消息类型，包括获取和设置字段的值，序列化你的消息到一个输出流中，从输入流中解析你的消息。

### 包

可以在`.proto` 文件中添加package说明符来避免协议消息类型键的名称冲突

### 定义服务

RPC（远程调用）系统中使用消息类型

可以在`.proto`文件中定义RPC服务接口，之后protocol buffer编译器会生成所选语言的服务接口代码和存根。

```
service SearchService {
  rpc Search (SearchRequest) returns (SearchResponse);
}
```

使用protocol buffer最直接的RPC系统是gRPC：由Google开发的，与语言和平台无关的开源RPC系统。gRPC与protocol buffer协同良好，它允许你使用特殊的protocol buffer插件直接从`.proto`文件中生成相关的RPC代码
