import time, random
from multiprocessing import Process, Pipe, current_process
from multiprocessing.connection import wait

def foo(w):
    for i in range(10):
        w.send((i, current_process().name))
    w.close()

if __name__ == '__main__':
    readers = []

    for i in range(4):
        r, w = Pipe(duplex=False)
        readers.append(r)
        p = Process(target=foo, args=(w,))
        p.start()
        # 이제 파이프의 쓰기 가능한 끝을 닫아서 p가 그 쓸 수 있는 핸들을 소유한 유일한
        # 프로세스가 되도록합니다. 이렇게하면 p가 쓰기 가능한 끝의 핸들을 닫을 때, 읽기
        # 가능한 끝의 wait() 가 즉시 준비가 되었다고 보고합니다.
        w.close()
    print(readers)

    while readers:
        a= []
        for r in wait(readers):
            try:
                msg = r.recv()
            except EOFError:
                readers.remove(r)
            else:
                a.append(msg)