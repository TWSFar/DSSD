import time


class Timer(object):
    def __init__(self):
        self.total_start_time = 0.
        self.epoch_start_time = 0.
        self.batch_start_time = 0.

    def total(self):
        cur = time.time()
        res = cur - self.total_start_time
        self.total_start_time = cur
        return res/3600

    def epoch(self):
        cur = time.time()
        res = cur - self.epoch_start_time
        self.epoch_start_time = cur
        return res/60

    def batch(self):
        cur = time.time()
        res = cur - self.batch_start_time
        self.batch_start_time = cur
        return res


if __name__ == "__main__":
    timer = Timer()
    timer.epoch()
    res = timer.epoch()
    print('%.2f' % res)
