from CISS_DemoPythonScript import CISSNode
import signal, time

def ctrl_c_handler(signal, frame, node):
    raise Exception("")

node = CISSNode()

signal.signal(signal.SIGINT, ctrl_c_handler)
print("\nData Streaming is started. Please see the log files...")
try:
    while 1:
        node.stream()
except Exception as e:
    node.disconnect()
    time.sleep(1)
    exit(0)