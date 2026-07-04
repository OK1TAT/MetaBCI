"""
WiFi EEG Relay — Python接收WiFi Shield数据，写到stdout给Java读
用法: python wifi_relay.py [host] [port]
"""
import socket, sys, time, json

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
LOCAL_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

def http(method, path, body=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect((HOST, 80))
    req = f"{method} {path} HTTP/1.1\r\nHost: {HOST}\r\nConnection: close\r\n"
    if body:
        req += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    req += "\r\n"
    s.sendall(req.encode())
    if body: s.sendall(body.encode())
    resp = b""
    while True:
        try:
            d = s.recv(4096)
            if not d: break
            resp += d
        except: break
    s.close()
    return resp.decode(errors="ignore").split("\r\n\r\n", 1)[-1]

# 猜本地IP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((HOST, 80))
local_ip = s.getsockname()[0]
s.close()

# 停止旧流
try: http("GET", "/stream/stop")
except: pass
try: http("DELETE", "/tcp")
except: pass

# 创建server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", LOCAL_PORT))
server.listen(1)
server.settimeout(1)

# 配置设备
tcp_payload = json.dumps({"ip": local_ip, "port": LOCAL_PORT, "output": "raw", "delimiter": False, "latency": 10000})
http("POST", "/tcp", tcp_payload)
http("GET", "/stream/start")

# 等待设备连接
deadline = time.time() + 20
sock = None
while time.time() < deadline:
    try:
        sock, peer = server.accept()
        break
    except socket.timeout:
        continue

if sock is None:
    sys.stderr.write("TIMEOUT: Device did not connect\n")
    sys.exit(1)

server.close()
sock.settimeout(1)

# 转发数据到stdout
while True:
    try:
        data = sock.recv(4096)
        if not data: break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except socket.timeout:
        continue
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        break
