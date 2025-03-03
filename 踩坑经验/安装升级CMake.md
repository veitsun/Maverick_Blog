# 方法一： 通过官方仓库安装

适用于 Ubuntu 、 Debian

```
# 卸载旧版本（如有必要）
sudo apt remove --purge cmake

# 添加 Kitware 的 APT 仓库
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update

# 安装 CMake
sudo apt install cmake
```

# 方法二：通过编译源码安装

```

```


