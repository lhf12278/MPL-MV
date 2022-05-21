import torch
import torch.nn as nn
from .metric import *


class CrossEntropyLabelEqual(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self,epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelEqual, self).__init__()
		# self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.ones(log_probs.size())/log_probs.size()[1]
		# targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu().long(), 1)
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self , epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelSmooth, self).__init__()

		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		# target = targets
		log_probs = self.logsoftmax(inputs)
		try:
			a = torch.zeros(log_probs.size())
			b = targets.unsqueeze(1).data.cpu().long()
			targets = a.scatter_(1, b, 1)
		except:
			print('erro')
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / log_probs.size()[1]
		loss = (- targets * log_probs).mean(0).sum()
		return loss


class T_EntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, epsilon=0.1, use_gpu=True):
		super(T_EntropyLabelSmooth, self).__init__()
		# self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		# self.logsoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		# target = targets
		probs = self.softmax(inputs)
		probs = torch.chunk(probs, 2, dim=1)
		probs = torch.cat((torch.sum(probs[0], dim=1).unsqueeze(1), torch.sum(probs[1], dim=1).unsqueeze(1)), 1)
		log_probs = torch.log(probs)
		# log_probs = self.logsoftmax(inputs)
		try:
			a = torch.zeros(log_probs.size())
			b = targets.unsqueeze(1).data.cpu().long()
			targets = a.scatter_(1, b, 1)
		except:
			print('erro')
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / log_probs.size()[1]
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class Camera_G_EntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, epsilon=0.1, use_gpu=True):
		super(Camera_G_EntropyLabelSmooth, self).__init__()
		# self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		# self.logsoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, targets,is_s=True):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		# target = targets
		probs = self.softmax(inputs)
		s_camera_probs = probs[:,:6]
		t_camera_probs = probs[:,6:]
		probs = torch.cat((torch.sum(s_camera_probs, dim=1).unsqueeze(1), torch.sum(t_camera_probs, dim=1).unsqueeze(1)), 1)
		log_probs = torch.log(probs)
		# log_probs = self.logsoftmax(inputs)
		try:
			a = torch.zeros(log_probs.size())
			b = targets.unsqueeze(1).data.cpu().long()
			targets = a.scatter_(1, b, 1)
		except:
			print('erro')
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / log_probs.size()[1]
		loss = (- targets * log_probs).mean(0).sum()
		# if is_s:
		# 	entropy_loss = (-s_camera_probs * (torch.log(s_camera_probs))).mean(0).sum()
		# 	loss = loss +entropy_loss
		# else:
		# 	entropy_loss = (-t_camera_probs * (torch.log(t_camera_probs))).mean(0).sum()
		# 	loss = loss + entropy_loss
		return loss

class Camera_D_EntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, epsilon=0.1, use_gpu=True):
		super(Camera_D_EntropyLabelSmooth, self).__init__()
		# self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		# self.logsoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, s_rate,s_label,t_label):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		# target = targets
		probs = self.softmax(inputs)
		# s_rate = label
		t_rate = 1 - s_rate

		# probs = torch.chunk(probs, 2, dim=1)
		# probs = torch.cat((torch.sum(probs[0], dim=1).unsqueeze(1), torch.sum(probs[1], dim=1).unsqueeze(1)), 1)
		log_probs = torch.log(probs)
		# log_probs = self.logsoftmax(inputs)

		ones = torch.ones(probs.size())
		s_label = torch.zeros(log_probs.size()).scatter_(1, s_label.unsqueeze(1).data.cpu().long(), 1) * (s_rate*ones)
		t_label = torch.zeros(log_probs.size()).scatter_(1, t_label.unsqueeze(1).data.cpu().long(), 1) * (t_rate*ones)
		targets = s_label + t_label

		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / log_probs.size()[1]
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class RankingLoss:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class TripletLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		if self.metric == 'cosine':
			mat_dist = cosine_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = cosine_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			mat_dist = euclidean_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = euclidean_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)

