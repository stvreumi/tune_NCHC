from multiprocessing.connection import Listener
from time import sleep
from datetime import timedelta, datetime
import pprint

pp = pprint.PrettyPrinter(indent=2)

def test_obf(hparam, num):
    x = hparam["x"]
    y = hparam["y"]
    sleep(3)
    return x**2 + abs(y) + y**3

def send_obj(address, authkey, obj, msg=None):
    with Listener(address, authkey=authkey) as listener:
        if msg != None:
            print(msg)
        else:
            print("send msg...")
        with listener.accept() as conn:
            conn.send(obj)
    print("send obj: ", obj)

def recv_obj(address, authkey, msg=None):
    with Listener(address, authkey=authkey) as listener:
        if msg != None:
            print(msg)
        else:
            print("recv_obj...")
        with listener.accept() as conn:
            return conn.recv()

def server_setup(address, authkey, obf):
    with Listener(address, authkey=authkey) as listener:
        print("server setup!")
        counter=0
        history = []
        while True:
            with listener.accept() as conn:
                counter+=1
                print("start #{}".format(counter))
                msg = conn.recv()
                if msg == "kill connection!":
                    conn.send(history)
                    return
                else:
                    print("hyperparameter:")
                    pp.pprint(msg)
                    start = datetime.now()
                    obf_value = obf(msg, counter)
                    end = datetime.now()
                    elapsed = end - start
                    conn.send("res")
                    conn.send(obf_value)
                    conn.send("info")
                    info = "end #{}, elapsed: {}\n".format(counter, str(elapsed))
                    conn.send(info)
                    # update history
                    msg["index"] = counter
                    msg["value"] = obf_value
                    msg["elapsed"] = str(elapsed)
                    history.append(msg)

if __name__ == "__main__":
    domain_socket = "/tmp/tuneconn"
    conn_authkey = b'physionet'
    send_obj(domain_socket, conn_authkey, "now is test")
    server_setup(domain_socket, conn_authkey, test_obf)