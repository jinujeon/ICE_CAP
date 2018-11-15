import threading, requests, time

class cam(threading.Thread):
    def __init__(self,location,id):
        # self.location = location
        # self.id = id
        threading.Thread.__init__(self)
        self.url = "http://127.0.0.1:8000/home/change_stat"
        self.count_warn = 0
        self.count_trash = 0
        self.is_warning = False
        self.is_trash = False
        self.trash_timer = 0
        self.e_list = []
        self.c_list = []
        self.data = {'cam_id': id, 'cam_status': 'safe', 'cam_location': location, 'trash': False}
        self.time = time.time()



class HtmlGetter():
    def __init__(self, url):
        # threading.Thread.__init__(self)
        self.url = url
        self.time = time.time()

    def run(self):
        resp = requests.get(self.url)
        time.sleep(1)
        print(self.url, len(resp.text), ' chars')
        a= c('1321132131223')
        print(a)
def c(s):
    return s
#
# a = HtmlGetter("http://naver.com")
# a.run()


class HtmlGetter_thread(threading.Thread):
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.url = url
        self.time = time.time()

    def run(self):
        resp = requests.get(self.url)
        time.sleep(1)
        print(self.url, len(resp.text), ' chars')
        print(time.time() - self.time)
        handle(self.url)

# a = time.time()
#
# for i in range(10):
#     i = HtmlGetter("http://naver.com")
#     i.run()
# print("time: {}".format(time.time() - a))
# a = time.time()
# c = []
# for i in range(10):
#     b = HtmlGetter_thread("http://google.com")
#     c.append(b.time)
#     b.start()
# print(c[9] - c[0])
# print("### End ###")

def handle(l):
    resp = requests.get(l.url)
    time.sleep(1)
    print(l.url, len(resp.text), ' chars')
    print(time.time() - l.time)

if __name__ == '__main__':
    info = ["Http://google.com","Http://naver.com","Http://mclab.hufs.ac.kr"]
    aa = time.time()
    for i in range(3):
        i = HtmlGetter_thread(info[i])
        i.start()