import torch
from tools import MultiItemAverageMeter, accuracy
from torch.cuda.amp import autocast
def c_label2cuda(label_list,base):
	for i in range(4):
		label_list[i][0] = label_list[i][0].to(base.device)
		label_list[i][1] = label_list[i][1].to(base.device)
	return label_list

def train_an_epoch(config, base, loaders, epoch=None,only_one=False):

	base.set_train()
	meter = MultiItemAverageMeter()

	### we assume 200 iterations as an epoch
	base.lr_scheduler.step(epoch)
	base.m_c1_scheduler.step(epoch)
	base.m_c2_scheduler.step(epoch)
	base.m_c3_scheduler.step(epoch)
	base.m_c4_scheduler.step(epoch)
	base.camera_scheduler.step(epoch)
	# base.d_2k_scheduler.step(epoch)
	# base.D_domainn_scheduler.step(epoch)
	if(epoch<config.start_epoch):
		steps = config.steps
	else:
		steps = config.steps_domain
	for i in range(steps):

		### load a batch data
		if(config.train_dataset[0]=='duke_market'):
			s_imgs, s_pids, s_cids,s_domain_label,s_pids_2k,_ = loaders.duke_train_iter.next_one()
			t_imgs, t_pids,t_cids,t_domain_label,t_pids_2k,_= loaders.market_train_iter.next_one()
			s_cids = (s_cids - torch.ones(s_cids.size())).int()
			t_cids = (t_cids - torch.ones(t_cids.size())).int()
			s_imgs, s_pids, s_domain_label, s_cids = s_imgs.to(base.device), s_pids.to(base.device), \
													 s_domain_label.to(base.device), s_cids.to(base.device)
			t_cids = (torch.ones(t_cids.size())*(config.s_camera_num)).int() + t_cids
			t_imgs, t_pids, t_domain_label, t_cids = t_imgs.to(base.device), t_pids.to(base.device), \
													 t_domain_label.to(base.device), t_cids.to(base.device)
			s_pids_2k = c_label2cuda(s_pids_2k, base)
			t_pids_2k = c_label2cuda(t_pids_2k, base)
			domain_label = (torch.ones(t_cids.size()) * 14).int()
			domain_label = domain_label.to(base.device)

		elif(config.train_dataset[0]=='market_duke'):
			s_imgs, s_pids, s_cids, s_domain_label, s_pids_2k, _ = loaders.market_train_iter.next_one()
			t_imgs, t_pids, t_cids, t_domain_label, t_pids_2k, _ = loaders.duke_train_iter.next_one()
			s_imgs, s_pids, s_domain_label, s_cids = s_imgs.to(base.device), s_pids.to(base.device), \
													 s_domain_label.to(base.device), s_cids.to(base.device)
			t_cids = (torch.ones(t_cids.size()) * (config.s_camera_num - 1)).int() + t_cids
			t_imgs, t_pids, t_domain_label, t_cids = t_imgs.to(base.device), t_pids.to(base.device), \
													 t_domain_label.to(base.device), t_cids.to(base.device)
			s_pids_2k = c_label2cuda(s_pids_2k, base)
			t_pids_2k = c_label2cuda(t_pids_2k, base)
			domain_label = (torch.ones(t_cids.size()) * 14).int()
			domain_label = domain_label.to(base.device)

		if config.baseline:
			features, cls_score = base.model(s_imgs,config.baseline)
			### loss
			ide_loss = base.ide_creiteron(cls_score, s_pids)
			# triplet_loss = base.triplet_creiteron(features, features, features, m_pids, m_pids, m_pids)
			# loss = ide_loss + triplet_loss
			acc = accuracy(cls_score, s_pids, [1])[0]
			### optimize
			if config.MultipleLoss:
				loss = loss/4
				loss.backward()
				if ((i + 1) % 4) == 0:
					# optimizer the net
					base.optimizer.step()  # update parameters of net
					base.optimizer.zero_grad()
			else:
				base.optimizer.zero_grad()
				ide_loss.backward()
				base.optimizer.step()
			### recored
			# meter.update({'ide_loss': ide_loss.data, 'triplet_loss': triplet_loss.data, 'acc': acc})
			meter.update({'ide_loss': ide_loss.data, 'acc': acc})
		else:
			if 'res' in config.cnnbackbone:
				if only_one:
					# if (epoch<config.start_epoch):
					# 	features, m_cls_score, _ = base.model(m_imgs, )
					# 	### loss
					# 	ide_loss = base.ide_creiteron(m_cls_score, m_pids)
					# 	triplet_loss = base.triplet_creiteron(features, features, features, m_pids, m_pids, m_pids)
					# 	loss = ide_loss + triplet_loss
					# 	acc = accuracy(m_cls_score, m_pids, [1])[0]
					# 	### optimize
					# 	if config.MultipleLoss:
					# 		loss = loss / 4
					# 		loss.backward()
					# 		if ((i + 1) % 4) == 0:
					# 			# optimizer the net
					# 			base.optimizer.step()  # update parameters of net
					# 			base.optimizer.zero_grad()
					# 	else:
					# 		base.optimizer.zero_grad()
					# 		loss.backward()
					# 		base.optimizer.step()
					# 	### recored
					# 	# meter.update({'ide_loss': ide_loss.data, 'triplet_loss': triplet_loss.data, 'acc': acc})
					# 	meter.update({'ide_loss': ide_loss.data, 'acc': acc})
					# else:
					# ### forward
					# # c_domain_id = []
					# # c_g_id = []
					# # for c_id in d_cids:
					# # 	if c_id <=4:
					# # 		c_domain_id.append(0)
					# # 		c_g_id.append(1)
					# # 	else:
					# # 		c_domain_id.append(1)
					# # 		c_g_id.append(0)
					# # c_domain_id = torch.tensor(c_domain_id).cuda()
					# # c_g_id = torch.tensor(c_g_id).cuda()
					# 	m_features, m_cls_score, m_features_list = base.model(m_imgs)
					# 	d_features, d_cls_score, d_features_list = base.model(d_imgs)
					# 	# s_domian_score = base.D_domian(m_features_list[0].detach())
					# 	# d_domian_score = base.D_domian(d_features_list[0].detach())
					# 	# 域鉴别器损失
					# 	# d_loss = base.ide_creiteron(s_domian_score, m_domain_label) + base.ide_creiteron(d_domian_score,
					# 	#                                                                                  d_domain_label)
					#
					# 	# if config.MultipleLoss:
					# 	# 	d_loss = d_loss / 4
					# 	# 	d_loss.backward()
					# 	# 	if ((i + 1) % 4) == 0:
					# 	# 		# optimizer the net
					# 	# 		base.D_domain_optimizer.step()  # update parameters of net
					# 	# 		base.D_domain_optimizer.zero_grad()
					# 	# 		torch.cuda.empty_cache()
					# 	# else:
					# 	# 	base.D_domain_optimizer.zero_grad()
					# 	# 	d_loss.backward()
					# 	# 	base.D_domain_optimizer.step()
					# 	# 	torch.cuda.empty_cache()
					# 	### 2k id分类器 market
					# 	s_c_id_score = base.m_c1_net(m_features_list[1].detach())
					# 	d_c_id_score = base.m_c1_net(d_features_list[1].detach())
					# 	m_cls_k_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[0][0])
					# 	d_cls_k_loss = base.t_c_creiteron(d_c_id_score,d_pids_2k[0][0])
					# 	c1_loss = m_cls_k_loss + d_cls_k_loss
					# 	if config.MultipleLoss:
					# 		m_cls_k_loss = m_cls_k_loss / 4
					# 		m_cls_k_loss.backward()
					# 		if ((i + 1) % 4) == 0:
					# 			# optimizer the net
					# 			base.m_2k_optimizer.step()  # update parameters of net
					# 			base.m_2k_optimizer.zero_grad()
					# 			torch.cuda.empty_cache()
					# 	else:
					# 		base.m_c1_optimizer.zero_grad()
					# 		c1_loss.backward()
					# 		base.m_c1_optimizer.step()
					# 		torch.cuda.empty_cache()
					#
					# 	s_c_id_score = base.m_c2_net(m_features_list[1].detach())
					# 	d_c_id_score = base.m_c2_net(d_features_list[1].detach())
					# 	m_cls_k_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[1][0])
					# 	d_cls_k_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[1][0])
					# 	c2_loss = m_cls_k_loss + d_cls_k_loss
					# 	base.m_c2_optimizer.zero_grad()
					# 	c2_loss.backward()
					# 	base.m_c2_optimizer.step()
					# 	torch.cuda.empty_cache()
					#
					# 	s_c_id_score = base.m_c3_net(m_features_list[1].detach())
					# 	d_c_id_score = base.m_c3_net(d_features_list[1].detach())
					# 	m_cls_k_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[2][0])
					# 	d_cls_k_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[2][0])
					# 	c3_loss = m_cls_k_loss + d_cls_k_loss
					# 	base.m_c3_optimizer.zero_grad()
					# 	c3_loss.backward()
					# 	base.m_c3_optimizer.step()
					# 	torch.cuda.empty_cache()
					#
					# 	s_c_id_score = base.m_c4_net(m_features_list[1].detach())
					# 	d_c_id_score = base.m_c4_net(d_features_list[1].detach())
					# 	m_cls_k_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[3][0])
					# 	d_cls_k_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[3][0])
					# 	c4_loss = m_cls_k_loss + d_cls_k_loss
					# 	base.m_c4_optimizer.zero_grad()
					# 	c4_loss.backward()
					# 	base.m_c4_optimizer.step()
					# 	torch.cuda.empty_cache()
					#
					# 	# s_domian_score = base.D_domian(m_features_list[0])
					# 	# d_domian_score = base.D_domian(d_features_list[0])
					# 	s_c_id_score = base.m_c1_net(m_features_list[1])
					# 	d_c_id_score = base.m_c1_net(d_features_list[1])
					# 	m_c1_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[0][1])
					# 	d_c1_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[0][1])
					# 	# d_c_id_score = base.d_camera_net(d_features_list[1])
					# 	# 域混淆
					# 	# d_confuse_loss = base.equal_creiteron(s_domian_score) + base.equal_creiteron(d_domian_score)
					# 	# 特征提取器
					# 	s_c_id_score = base.m_c2_net(m_features_list[1])
					# 	d_c_id_score = base.m_c2_net(d_features_list[1])
					# 	m_c2_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[1][1])
					# 	d_c2_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[1][1])
					#
					# 	s_c_id_score = base.m_c3_net(m_features_list[1])
					# 	d_c_id_score = base.m_c3_net(d_features_list[1])
					# 	m_c3_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[2][1])
					# 	d_c3_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[2][1])
					# 	# d_cls_2k_loss = base.ide_creiteron(d_c_id_score, d_pids_2k)
					# 	# cls_2k_loss = m_cls_2k_loss
					# 	s_c_id_score = base.m_c4_net(m_features_list[1])
					# 	d_c_id_score = base.m_c4_net(d_features_list[1])
					# 	m_c4_loss = base.ide_creiteron(s_c_id_score, m_pids_2k[3][1])
					# 	d_c4_loss = base.t_c_creiteron(d_c_id_score, d_pids_2k[3][1])
					#
					# 	m_ide_loss = base.ide_creiteron(m_cls_score, m_pids)
					# 	m_triplet_loss = base.triplet_creiteron(m_features, m_features, m_features, m_pids, m_pids, m_pids)
					# 	m_loss = m_ide_loss + m_triplet_loss + m_c1_loss+d_c1_loss+m_c2_loss+d_c2_loss+m_c3_loss+d_c3_loss+m_c4_loss+d_c4_loss
					# 	acc = accuracy(m_cls_score, m_pids, [1])[0]
					# 	### optimize
					# 	if config.MultipleLoss:
					# 		m_loss = m_loss / 4
					# 		m_loss.backward()
					# 		if ((i + 1) % 4) == 0:
					# 			# optimizer the net
					# 			base.opttuxiangrtuimizer.step()  # update parameters of net
					# 			base.optimizer.zero_grad()
					# 			torch.cuda.empty_cache()
					# 	else:
					# 		base.optimizer.zero_grad()
					# 		m_loss.backward()
					# 		base.optimizer.step()
					# 		torch.cuda.empty_cache()
					#
					# 	### recored
					# 	meter.update({'ide_loss': m_ide_loss.data,
					# 	              'c1_d': c1_loss, 'c1_g': m_c1_loss+d_c1_loss,'c2_d': c2_loss, 'c2_g': m_c2_loss+d_c2_loss,
					# 	              'c3_d':c3_loss,'c3_g':m_c3_loss+d_c3_loss,'c4_d':c4_loss,'c4_g':m_c4_loss+d_c4_loss,
					# 	              'acc': acc})
					print('1')
				else:
					### forward
					if (epoch<config.start_epoch):
						merge_feature, s_cls_score,merge_score,s_features_list,_ = base.model(s_imgs)
						### loss
						s_ide_loss = base.ide_creiteron(s_cls_score[0], s_pids) + base.ide_creiteron(s_cls_score[1],s_pids) \
						             + base.ide_creiteron(merge_score,s_pids)
						s_triplet_loss = base.triplet_creiteron(s_features_list[0], s_features_list[0],
																s_features_list[0], s_pids, s_pids, s_pids) + \
										 base.triplet_creiteron(s_features_list[1], s_features_list[1],
																s_features_list[1], s_pids, s_pids, s_pids) + \
										 base.triplet_creiteron(merge_feature,merge_feature,merge_feature,s_pids, s_pids, s_pids)
						s_loss = s_ide_loss + s_triplet_loss
						acc = accuracy(merge_score, s_pids, [1])[0]

						base.optimizer.zero_grad()
						s_loss.backward()
						base.optimizer.step()
						torch.cuda.empty_cache()
						meter.update({'ide_loss': s_ide_loss.data, 'triplet_loss':s_triplet_loss.data ,'acc': acc})
					else:

						merge_feature, s_cls_score,merge_score,s_features_list,s_max_middle = base.model(s_imgs)

						_, _,_, t_features_list,t_max_middle = base.model(t_imgs)
						s_t_max_middle = [s_max_middle,t_max_middle]
						mix_feature,label_a = base.model(s_imgs,s_t_max_middle,mix=True)
						# ### 域鉴别器
						# s_domian_score = base.D_domian(m_features_list[0].detach())
						# d_domian_score = base.D_domian(d_features_list[0].detach())
						# mix_domain_score = base.D_domian(mix_feature.detach())
						# #域鉴别器损失
						# mix_domain_label = d_domain_label + d_domain_label
						# d_domain_loss = base.ide_creiteron(d_domian_score,d_domain_label)
						# d_loss = base.ide_creiteron(s_domian_score,m_domain_label)+ d_domain_loss\
						#          + base.ide_creiteron(mix_domain_score,mix_domain_label)
						#
						# base.D_domain_optimizer.zero_grad()
						# d_loss.backward()
						# base.D_domain_optimizer.step()
						# torch.cuda.empty_cache()

						#相机分类器
						if (epoch < 80):
							s_camera_score = base.carmera_net(s_features_list[0].detach())
							t_camera_score = base.carmera_net(t_features_list[0].detach())
							mix_camera_score = base.carmera_net(mix_feature.detach())
							camera_d_t_loss = base.ide_creiteron(t_camera_score, t_cids)
							camera_d_loss = base.ide_creiteron(s_camera_score, s_cids) + camera_d_t_loss \
											+ base.camera_d_creiteron(mix_camera_score, label_a, s_cids, t_cids)
							camera_d_loss = config.m * camera_d_loss
							base.camera_optimizer.zero_grad()
							camera_d_loss.backward()
							base.camera_optimizer.step()
							torch.cuda.empty_cache()
							# s_domian_score = base.D_domian(s_features_list[0].detach())
							# t_domian_score = base.D_domian(t_features_list[0].detach())
							# mix_domain_score = base.D_domian(mix_feature.detach())
							# # 域鉴别器损失
							# # mix_domain_label = d_domain_label + d_domain_label
							# d_domain_loss = base.ide_creiteron(t_domian_score, t_domain_label)
							# d_loss = base.ide_creiteron(s_domian_score, s_domain_label) + d_domain_loss \
							# 		 + base.camera_d_creiteron(mix_domain_score, label_a, s_domain_label,
							# 								   t_domain_label)
							#
							# base.D_domain_optimizer.zero_grad()
							# d_loss.backward()
							# base.D_domain_optimizer.step()
							# torch.cuda.empty_cache()

							s_camera_score = base.carmera_net(s_features_list[0])
							t_camera_score = base.carmera_net(t_features_list[0])
							mix_camera_score = base.carmera_net(mix_feature)
							t_camera_g_loss = base.ide_creiteron(t_camera_score, domain_label)
							camera_g_loss = base.ide_creiteron(s_camera_score,domain_label) + t_camera_g_loss + base.ide_creiteron(
								mix_camera_score, domain_label)
							# s_domian_score = base.D_domian(s_features_list[0])
							# d_domian_score = base.D_domian(t_features_list[0])
							# mix_domain_score = base.D_domian(mix_feature)
							# # 域混淆
							# confuse_label = t_domain_label + t_domain_label
							# t_d_confuse_loss = base.ide_creiteron(d_domian_score, confuse_label)
							# mix_domain_loss = base.ide_creiteron(mix_domain_score, confuse_label)
							# d_confuse_loss = base.ide_creiteron(s_domian_score,confuse_label) + t_d_confuse_loss + mix_domain_loss

							s_ide_loss = base.ide_creiteron(s_cls_score[0], s_pids) + base.ide_creiteron(s_cls_score[1],
																										 s_pids) \
										 + base.ide_creiteron(merge_score, s_pids)
							s_triplet_loss = base.triplet_creiteron(s_features_list[0], s_features_list[0],
																	s_features_list[0], s_pids, s_pids, s_pids) + \
											 base.triplet_creiteron(s_features_list[1], s_features_list[1],
																	s_features_list[1], s_pids, s_pids, s_pids) + \
											 base.triplet_creiteron(merge_feature, merge_feature, merge_feature, s_pids, s_pids, s_pids)
							s_loss = s_ide_loss + s_triplet_loss  + config.m * camera_g_loss
							# m_loss = m_ide_loss + m_triplet_loss + camera_g_loss
							acc = accuracy(s_cls_score[0], s_pids, [1])[0]
							### optimize

							base.optimizer.zero_grad()
							s_loss.backward()
							base.optimizer.step()
							torch.cuda.empty_cache()

							### recored
							meter.update({'ide_loss': s_ide_loss.data,
										  # 'Two_T_D:': d1_cls_k_loss + d2_cls_k_loss + d3_cls_k_loss + d4_cls_k_loss,
										  # 'Two_T_G': d_c1_loss + d_c2_loss + d_c3_loss + d_c4_loss,
										  'One_T1_D': camera_d_loss, 'One_T1_G': camera_g_loss,
										  # 'One_T2_D': d_domain_loss, 'One_T2_G': t_d_confuse_loss,
										  'acc': acc})
						else:
							s_camera_score = base.carmera_net(s_features_list[0].detach())
							t_camera_score = base.carmera_net(t_features_list[0].detach())
							mix_camera_score = base.carmera_net(mix_feature.detach())
							camera_d_t_loss = base.ide_creiteron(t_camera_score,t_cids)
							camera_d_loss = base.ide_creiteron(s_camera_score,s_cids)+camera_d_t_loss \
											+ base.camera_d_creiteron(mix_camera_score,label_a,s_cids,t_cids)
							base.camera_optimizer.zero_grad()
							camera_d_loss.backward()
							base.camera_optimizer.step()
							torch.cuda.empty_cache()
							# s_domian_score = base.D_domian(s_features_list[0].detach())
							# t_domian_score = base.D_domian(t_features_list[0].detach())
							# mix_domain_score = base.D_domian(mix_feature.detach())
							# # 域鉴别器损失
							# # mix_domain_label = d_domain_label + d_domain_label
							# d_domain_loss = base.ide_creiteron(t_domian_score, t_domain_label)
							# d_loss = base.ide_creiteron(s_domian_score, s_domain_label) + d_domain_loss \
							# 		 + base.camera_d_creiteron(mix_domain_score, label_a, s_domain_label,
							# 								   t_domain_label)
							#
							# base.D_domain_optimizer.zero_grad()
							# d_loss.backward()
							# base.D_domain_optimizer.step()
							# torch.cuda.empty_cache()

							s_camera_score = base.carmera_net(s_features_list[0])
							t_camera_score = base.carmera_net(t_features_list[0])
							mix_camera_score = base.carmera_net(mix_feature)
							t_camera_g_loss = base.ide_creiteron(t_camera_score, domain_label)
							camera_g_loss = base.ide_creiteron(s_camera_score,domain_label) + t_camera_g_loss + base.ide_creiteron(
								mix_camera_score, domain_label)
							camera_g_loss = config.m * camera_g_loss


							s_c_id_score = base.m_c1_net(s_features_list[1].detach())
							t_c_id_score = base.m_c1_net(t_features_list[1].detach())
							s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[0][0])
							t1_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[0][0])
							c1_loss = config.h*(s_cls_k_loss + t1_cls_k_loss)
							base.m_c1_optimizer.zero_grad()
							c1_loss.backward()
							base.m_c1_optimizer.step()
							torch.cuda.empty_cache()

							s_c_id_score = base.m_c2_net(s_features_list[1].detach())
							t_c_id_score = base.m_c2_net(t_features_list[1].detach())
							m_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[1][0])
							t2_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[1][0])
							c2_loss = config.h*(m_cls_k_loss + t2_cls_k_loss)
							base.m_c2_optimizer.zero_grad()
							c2_loss.backward()
							base.m_c2_optimizer.step()
							torch.cuda.empty_cache()

							s_c_id_score = base.m_c3_net(s_features_list[1].detach())
							t_c_id_score = base.m_c3_net(t_features_list[1].detach())
							s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[2][0])
							t3_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[2][0])
							c3_loss = config.h*(s_cls_k_loss + t3_cls_k_loss)
							base.m_c3_optimizer.zero_grad()
							c3_loss.backward()
							base.m_c3_optimizer.step()
							torch.cuda.empty_cache()

							s_c_id_score = base.m_c4_net(s_features_list[1].detach())
							t_c_id_score = base.m_c4_net(t_features_list[1].detach())
							s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[3][0])
							t4_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[3][0])
							c4_loss = config.h*(s_cls_k_loss + t4_cls_k_loss)
							base.m_c4_optimizer.zero_grad()
							c4_loss.backward()
							base.m_c4_optimizer.step()
							torch.cuda.empty_cache()

							# s_domian_score = base.D_domian(m_features_list[0].detach())
							# d_domian_score = base.D_domian(d_features_list[0].detach())
							#
							#
							# s_domian_score = base.D_domian(m_features_list[0])
							# d_domian_score = base.D_domian(d_features_list[0])
							# 域混淆
							# confuse_label = d_domain_label + mix_domain_label
							# t_d_confuse_loss = base.ide_creiteron(d_domian_score, confuse_label)
							# d_confuse_loss = base.ide_creiteron(s_domian_score, confuse_label) + t_d_confuse_loss

							# s_camera_score = base.carmera_net(s_features_list[0])
							# t_camera_score = base.carmera_net(t_features_list[0])
							# mix_camera_score = base.carmera_net(mix_feature)
							# t_camera_g_loss = base.ide_creiteron(t_camera_score,domain_label)
							# camera_g_loss = base.ide_creiteron(s_camera_score, domain_label) + t_camera_g_loss + base.ide_creiteron(mix_camera_score,domain_label)
							# s_domian_score = base.D_domian(s_features_list[0])
							# d_domian_score = base.D_domian(t_features_list[0])
							# mix_domain_score = base.D_domian(mix_feature)
							# # 域混淆
							# confuse_label = t_domain_label + t_domain_label
							# t_d_confuse_loss = base.ide_creiteron(d_domian_score, confuse_label)
							# mix_domain_loss = base.ide_creiteron(mix_domain_score, confuse_label)
							# d_confuse_loss = base.ide_creiteron(s_domian_score,
							# 									confuse_label) + t_d_confuse_loss + mix_domain_loss

							s_c_id_score = base.m_c1_net(s_features_list[1])
							t_c_id_score = base.m_c1_net(t_features_list[1])
							s_c1_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[0][1])
							t_c1_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[0][1])
							# d_c_id_score = base.d_camera_net(d_features_list[1])

							# 特征提取器
							s_c_id_score = base.m_c2_net(s_features_list[1])
							t_c_id_score = base.m_c2_net(t_features_list[1])
							s_c2_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[1][1])
							t_c2_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[1][1])

							s_c_id_score = base.m_c3_net(s_features_list[1])
							t_c_id_score = base.m_c3_net(t_features_list[1])
							s_c3_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[2][1])
							t_c3_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[2][1])
							# d_cls_2k_loss = base.ide_creiteron(d_c_id_score, d_pids_2k)
							# cls_2k_loss = m_cls_2k_loss
							s_c_id_score = base.m_c4_net(s_features_list[1])
							t_c_id_score = base.m_c4_net(t_features_list[1])
							s_c4_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[3][1])
							t_c4_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[3][1])

							s_ide_loss = base.ide_creiteron(s_cls_score[0], s_pids)+base.ide_creiteron(s_cls_score[1], s_pids)\
										 + base.ide_creiteron(merge_score,s_pids)
							s_triplet_loss = base.triplet_creiteron(s_features_list[0], s_features_list[0], s_features_list[0], s_pids, s_pids, s_pids)+\
											 base.triplet_creiteron(s_features_list[1], s_features_list[1], s_features_list[1], s_pids, s_pids, s_pids)+ \
											 base.triplet_creiteron(merge_feature, merge_feature, merge_feature, s_pids, s_pids, s_pids)
							# if(epoch>88):
							# 	m_loss = s_ide_loss + s_triplet_loss + 0.1 * (camera_g_loss + s_c1_loss + t_c1_loss + s_c2_loss + t_c2_loss + s_c3_loss + t_c3_loss + s_c4_loss + t_c4_loss)
							# else:
							m_loss = s_ide_loss + s_triplet_loss \
							         + config.m * camera_g_loss \
							         + config.h*(s_c1_loss + t_c1_loss + s_c2_loss + t_c2_loss + s_c3_loss + t_c3_loss + s_c4_loss + t_c4_loss)
							# m_loss = m_ide_loss + m_triplet_loss + camera_g_loss
							acc = accuracy(s_cls_score[0], s_pids, [1])[0]
							### optimize

							base.optimizer.zero_grad()
							m_loss.backward()
							base.optimizer.step()
							torch.cuda.empty_cache()

							### recored
							meter.update({'ide_loss': s_ide_loss.data,
										  'Two_T_D:': t1_cls_k_loss+t2_cls_k_loss+t3_cls_k_loss+t4_cls_k_loss, 'Two_T_G': t_c1_loss+t_c2_loss+t_c3_loss+t_c4_loss,
										  'One_T1_D': camera_d_loss,'One_T1_G': camera_g_loss,
										  # 'One_T2_D': d_domain_loss, 'One_T2_G': t_d_confuse_loss,
										  'acc': acc})


	return meter.get_val(), meter.get_str()
