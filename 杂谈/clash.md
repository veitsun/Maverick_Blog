# 查看日志

tail -f ~/.config/clash/clash.log

# 测试是否连通

curl -x http://127.0.0.1:7890 https://www.google.com

其中 ping 不走代理（只设置了 http 和 https 代理）

http_proxy="http://127.0.0.1:7890"  
https_proxy="http://127.0.0.1:7890"  
all_proxy="socks5://127.0.0.1:7890"
