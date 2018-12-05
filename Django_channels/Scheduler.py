import requests
import json
class Frame_scheduler:
    def __init__(self,id_list,fps):
        self.detection_weight = []
        self.cam_count = len(id_list)
        self.id_list = id_list
        self.schedule = []
        # self.cam_stat_dict = dict([(id, None) for id in range(self.cam_count)])
        self.bufsize = fps

    def set_cam_frame_order(self, option=None):
        if option is None:## default ##
            self.schedule = []#initialize
            self.bufsize = 10
            for i in range(self.bufsize):
                self.schedule.append(self.id_list[(i % self.cam_count)])
        else:
            self.detection_weight = self.get_cam_weight()
            print("CAMS WEIGHT: ",self.detection_weight)
            self.schedule = []#initialize
            sum = 0
            for i in self.detection_weight:
                sum += i
            self.bufsize = sum
            i = 0
            while len(self.schedule) < self.bufsize:
                print("while loop ")
                if self.detection_weight[self.id_list[(i % self.cam_count)]] != 0:
                    self.schedule.append(self.id_list[(i % self.cam_count)])
                    self.detection_weight[self.id_list[(i % self.cam_count)]] -= 1
                i += 1

    def get_cam_weight(self):
        detection_weight = []
        url = 'http://127.0.0.1:8000/home/send_weight'
        response = requests.get(url=url)
        decoded = response.content.decode('UTF-8')
        weight_dict = json.loads(decoded)
        for id in range(self.cam_count):
            detection_weight.append(weight_dict[str(id)])
        return detection_weight