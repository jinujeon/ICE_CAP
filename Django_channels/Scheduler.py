class Frame_scheduler:
    def __init__(self,detection_weigth,id_list,fps):
        self.detection_weight = detection_weight
        self.cam_count = len(id_list)
        self.id_list = id_list
        self.schedule = []
        # self.cam_stat_dict = dict([(id, None) for id in range(self.cam_count)])
        self.bufsize = fps

    def set_cam_frame_order(self, option = None):
        if option is None:## default ##
            self.schedule = []#initialize
            self.bufsize = 10
            for i in range(self.bufsize):
                self.schedule.append(self.id_list[(i % self.cam_count)])
        else:
            self.schedule = []#initialize
            sum = 0
            for i in self.detection_weight:
                sum += i
            self.bufsize = sum
            i = 0
            while len(self.schedule) < self.bufsize:
                if self.detection_weight[self.id_list[(i % self.cam_count)]] != 0:
                    self.schedule.append(self.id_list[(i % self.cam_count)])
                    self.detection_weight[self.id_list[(i % self.cam_count)]] -= 1
                i += 1


max_fps = 10
cam_id_list = [0,1]
detection_weight = [9,9]
schedule = Frame_scheduler(detection_weight, cam_id_list ,max_fps)
schedule.set_cam_frame_order(1)
print(schedule.schedule)
