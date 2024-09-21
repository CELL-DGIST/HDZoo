import time
import torch

from collections import defaultdict


""" Timing Collector  """
class TimingCollector:
	def __init__(self):
		self.reset()

	def __del__(self):
		pass

	def reset(self):
		self.etime_per_tag = defaultdict(int)
		self.count_per_tag = defaultdict(int)

	def print(self):
		print(str(self))

	def __str__(self):
		s = ""
		for tag in self.etime_per_tag.keys():
			total = self.etime_per_tag[tag]
			avg = total / self.count_per_tag[tag]
			s += "{}\t{}\t{}\n".format(tag, total, avg)
		return (s)

	def add(self, tag, elasped_time):
		self.etime_per_tag[tag] += elasped_time
		self.count_per_tag[tag] += 1

	@staticmethod
	def g_instance():  # Global instance for singleton-like impl.
		if TimingCollector.instance is None:
			TimingCollector.instance = TimingCollector()
		return TimingCollector.instance
TimingCollector.instance = None


class Timing:
    def __init__(self, tag, sync_cuda=True, timing_collector=None):
        self.tag = tag
        self.sync_cuda = sync_cuda
        if timing_collector is None:
            self.collector = TimingCollector.g_instance()
        else:
            self.collector = timing_collector

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()

        self.start = time.perf_counter()

    def __exit__(self, t, v, tb):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.collector.add(
                self.tag,
                time.perf_counter() - self.start)
