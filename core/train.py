import torch
from tools import MultiItemAverageMeter, accuracy
from torch.cuda.amp import autocast
def c_label2cuda(label_list,base,num):
	for i in range(num):
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
		if(config.train_task=='duke_market' or config.train_task=='market_duke'):
			s_imgs, s_pids, s_cids,s_domain_label,s_pids_2k,_ = loaders.s_train_iter.next_one()
			t_imgs, t_pids,t_cids,t_domain_label,t_pids_2k,_= loaders.t_train_iter.next_one()
			s_cids = (s_cids - torch.ones(s_cids.size())).int()
			t_cids = (t_cids - torch.ones(t_cids.size())).int()
			s_imgs, s_pids, s_domain_label, s_cids = s_imgs.to(base.device), s_pids.to(base.device), \
													 s_domain_label.to(base.device), s_cids.to(base.device)

			t_cids = (torch.ones(t_cids.size())*(config.s_camera_num)).int() + t_cids
			t_imgs, t_pids, t_domain_label, t_cids = t_imgs.to(base.device), t_pids.to(base.device), \
													 t_domain_label.to(base.device), t_cids.to(base.device)
			s_pids_2k = c_label2cuda(s_pids_2k, base,4)
			t_pids_2k = c_label2cuda(t_pids_2k, base,4)

			domain_label = (torch.ones(t_cids.size()) * 14).int()
			domain_label = domain_label.to(base.device)

		elif('msmt' in config.train_task):
			base.m_c5_scheduler.step(epoch)
			base.m_c6_scheduler.step(epoch)
			base.m_c7_scheduler.step(epoch)
			base.m_c8_scheduler.step(epoch)
			s_imgs, s_pids, s_cids, s_domain_label, s_pids_2k, _ = loaders.s_train_iter.next_one()
			t_imgs, t_pids, t_cids, t_domain_label, t_pids_2k, _ = loaders.t_train_iter.next_one()
			s_cids = (s_cids - torch.ones(t_cids.size())).int()
			s_imgs, s_pids, s_domain_label, s_cids = s_imgs.to(base.device), s_pids.to(base.device), \
													 s_domain_label.to(base.device), s_cids.to(base.device)

			t_cids = (torch.ones(t_cids.size()) * (config.s_camera_num)).int() + t_cids
			t_imgs, t_pids, t_domain_label, t_cids = t_imgs.to(base.device), t_pids.to(base.device), \
													 t_domain_label.to(base.device), t_cids.to(base.device)
			s_pids_2k = c_label2cuda(s_pids_2k, base,8)
			t_pids_2k = c_label2cuda(t_pids_2k, base,8)

			if config.train_task=='duke_msmt' or config.train_task=='msmt_duke':
				domain_label = (torch.ones(t_cids.size()) * 23).int()
			elif config.train_task=='market_msmt' or config.train_task=='msmt_market':
				domain_label = (torch.ones(t_cids.size()) * 21).int()
			domain_label = domain_label.to(base.device)

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

			#相机分类器
			if (epoch < 80):
				s_camera_score = base.carmera_net(s_features_list[0].detach())
				t_camera_score = base.carmera_net(t_features_list[0].detach())
				mix_camera_score = base.carmera_net(mix_feature.detach())
				camera_d_t_loss = base.ide_creiteron(t_camera_score, t_cids)
				camera_d_loss = base.ide_creiteron(s_camera_score, s_cids) + camera_d_t_loss \
								+ base.camera_d_creiteron(mix_camera_score, label_a, s_cids, t_cids)
				base.camera_optimizer.zero_grad()
				camera_d_loss.backward()
				base.camera_optimizer.step()
				torch.cuda.empty_cache()
				# # 域鉴别器损失

				s_camera_score = base.carmera_net(s_features_list[0])
				t_camera_score = base.carmera_net(t_features_list[0])
				mix_camera_score = base.carmera_net(mix_feature)
				t_camera_g_loss = base.ide_creiteron(t_camera_score, domain_label)
				camera_g_loss = base.ide_creiteron(s_camera_score,domain_label) + t_camera_g_loss + base.ide_creiteron(
					mix_camera_score, domain_label)
				# # 域混淆

				s_ide_loss = base.ide_creiteron(s_cls_score[0], s_pids) + base.ide_creiteron(s_cls_score[1],
																							 s_pids) \
							 + base.ide_creiteron(merge_score, s_pids)
				s_triplet_loss = base.triplet_creiteron(s_features_list[0], s_features_list[0],
														s_features_list[0], s_pids, s_pids, s_pids) + \
								 base.triplet_creiteron(s_features_list[1], s_features_list[1],
														s_features_list[1], s_pids, s_pids, s_pids) + \
								 base.triplet_creiteron(merge_feature, merge_feature, merge_feature, s_pids, s_pids, s_pids)
				s_loss = s_ide_loss + s_triplet_loss  + camera_g_loss
				acc = accuracy(s_cls_score[0], s_pids, [1])[0]
				### optimize

				base.optimizer.zero_grad()
				s_loss.backward()
				base.optimizer.step()
				torch.cuda.empty_cache()

				### recored
				meter.update({'ide_loss': s_ide_loss.data,
							  'One_T1_D': camera_d_loss, 'One_T1_G': camera_g_loss,
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

				s_camera_score = base.carmera_net(s_features_list[0])
				t_camera_score = base.carmera_net(t_features_list[0])
				mix_camera_score = base.carmera_net(mix_feature)
				t_camera_g_loss = base.ide_creiteron(t_camera_score, domain_label)
				camera_g_loss = base.ide_creiteron(s_camera_score,domain_label) + t_camera_g_loss + base.ide_creiteron(
					mix_camera_score, domain_label)


				###########################
				s_c_id_score = base.m_c1_net(s_features_list[1].detach())
				t_c_id_score = base.m_c1_net(t_features_list[1].detach())
				s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[0][0])
				t1_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[0][0])
				c1_loss = s_cls_k_loss + t1_cls_k_loss
				base.m_c1_optimizer.zero_grad()
				c1_loss.backward()
				base.m_c1_optimizer.step()
				torch.cuda.empty_cache()

				s_c_id_score = base.m_c2_net(s_features_list[1].detach())
				t_c_id_score = base.m_c2_net(t_features_list[1].detach())
				m_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[1][0])
				t2_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[1][0])
				c2_loss = m_cls_k_loss + t2_cls_k_loss
				base.m_c2_optimizer.zero_grad()
				c2_loss.backward()
				base.m_c2_optimizer.step()
				torch.cuda.empty_cache()

				s_c_id_score = base.m_c3_net(s_features_list[1].detach())
				t_c_id_score = base.m_c3_net(t_features_list[1].detach())
				s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[2][0])
				t3_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[2][0])
				c3_loss = s_cls_k_loss + t3_cls_k_loss
				base.m_c3_optimizer.zero_grad()
				c3_loss.backward()
				base.m_c3_optimizer.step()
				torch.cuda.empty_cache()

				s_c_id_score = base.m_c4_net(s_features_list[1].detach())
				t_c_id_score = base.m_c4_net(t_features_list[1].detach())
				s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[3][0])
				t4_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[3][0])
				c4_loss = s_cls_k_loss + t4_cls_k_loss
				base.m_c4_optimizer.zero_grad()
				c4_loss.backward()
				base.m_c4_optimizer.step()
				torch.cuda.empty_cache()

				if 'msmt' in  config.train_task:
					s_c_id_score = base.m_c5_net(s_features_list[1].detach())
					t_c_id_score = base.m_c5_net(t_features_list[1].detach())
					s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[4][0])
					t5_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[4][0])
					c5_loss = s_cls_k_loss + t5_cls_k_loss
					base.m_c5_optimizer.zero_grad()
					c5_loss.backward()
					base.m_c5_optimizer.step()
					torch.cuda.empty_cache()

					s_c_id_score = base.m_c6_net(s_features_list[1].detach())
					t_c_id_score = base.m_c6_net(t_features_list[1].detach())
					s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[5][0])
					t6_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[5][0])
					c6_loss = s_cls_k_loss + t6_cls_k_loss
					base.m_c6_optimizer.zero_grad()
					c6_loss.backward()
					base.m_c6_optimizer.step()
					torch.cuda.empty_cache()

					s_c_id_score = base.m_c7_net(s_features_list[1].detach())
					t_c_id_score = base.m_c7_net(t_features_list[1].detach())
					s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[6][0])
					t7_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[6][0])
					c7_loss = s_cls_k_loss + t7_cls_k_loss
					base.m_c7_optimizer.zero_grad()
					c7_loss.backward()
					base.m_c7_optimizer.step()
					torch.cuda.empty_cache()

					s_c_id_score = base.m_c8_net(s_features_list[1].detach())
					t_c_id_score = base.m_c8_net(t_features_list[1].detach())
					s_cls_k_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[7][0])
					t8_cls_k_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[7][0])
					c8_loss = s_cls_k_loss + t8_cls_k_loss
					base.m_c8_optimizer.zero_grad()
					c8_loss.backward()
					base.m_c8_optimizer.step()
					torch.cuda.empty_cache()

				###############################################
				s_c_id_score = base.m_c1_net(s_features_list[1])
				t_c_id_score = base.m_c1_net(t_features_list[1])
				s_c1_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[0][1])
				t_c1_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[0][1])

				s_c_id_score = base.m_c2_net(s_features_list[1])
				t_c_id_score = base.m_c2_net(t_features_list[1])
				s_c2_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[1][1])
				t_c2_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[1][1])

				s_c_id_score = base.m_c3_net(s_features_list[1])
				t_c_id_score = base.m_c3_net(t_features_list[1])
				s_c3_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[2][1])
				t_c3_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[2][1])


				s_c_id_score = base.m_c4_net(s_features_list[1])
				t_c_id_score = base.m_c4_net(t_features_list[1])
				s_c4_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[3][1])
				t_c4_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[3][1])

				if 'msmt' in config.train_task:
					s_c_id_score = base.m_c5_net(s_features_list[1])
					t_c_id_score = base.m_c5_net(t_features_list[1])
					s_c5_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[4][1])
					t_c5_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[4][1])

					s_c_id_score = base.m_c6_net(s_features_list[1])
					t_c_id_score = base.m_c6_net(t_features_list[1])
					s_c6_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[5][1])
					t_c6_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[5][1])

					s_c_id_score = base.m_c7_net(s_features_list[1])
					t_c_id_score = base.m_c7_net(t_features_list[1])
					s_c7_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[6][1])
					t_c7_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[6][1])

					s_c_id_score = base.m_c8_net(s_features_list[1])
					t_c_id_score = base.m_c8_net(t_features_list[1])
					s_c8_loss = base.ide_creiteron(s_c_id_score, s_pids_2k[7][1])
					t_c8_loss = base.t_c_creiteron(t_c_id_score, t_pids_2k[7][1])

				s_ide_loss = base.ide_creiteron(s_cls_score[0], s_pids)+base.ide_creiteron(s_cls_score[1], s_pids)\
							 + base.ide_creiteron(merge_score,s_pids)
				s_triplet_loss = base.triplet_creiteron(s_features_list[0], s_features_list[0], s_features_list[0], s_pids, s_pids, s_pids)+\
								 base.triplet_creiteron(s_features_list[1], s_features_list[1], s_features_list[1], s_pids, s_pids, s_pids)+ \
								 base.triplet_creiteron(merge_feature, merge_feature, merge_feature, s_pids, s_pids, s_pids)

				if 'msmt' in config.train_task:
					m_loss = s_ide_loss + s_triplet_loss + camera_g_loss \
						   + s_c1_loss + t_c1_loss + s_c2_loss + t_c2_loss + s_c3_loss + t_c3_loss + s_c4_loss + t_c4_loss \
						   + s_c5_loss + t_c5_loss + s_c6_loss + t_c6_loss + s_c7_loss + t_c7_loss + s_c8_loss + t_c8_loss
				else:
					m_loss = s_ide_loss + s_triplet_loss + camera_g_loss \
							 + s_c1_loss + t_c1_loss + s_c2_loss + t_c2_loss + s_c3_loss + t_c3_loss + s_c4_loss + t_c4_loss

				acc = accuracy(s_cls_score[0], s_pids, [1])[0]
				### optimize

				base.optimizer.zero_grad()
				m_loss.backward()
				base.optimizer.step()
				torch.cuda.empty_cache()

				### recored
				meter.update({'ide_loss': s_ide_loss.data,
							  'Two_T_D:': t1_cls_k_loss+t2_cls_k_loss+t3_cls_k_loss+t4_cls_k_loss, 'Two_T_G': t_c1_loss+t_c2_loss+t_c3_loss+t_c4_loss,
							  'One_T_D': camera_d_loss,'One_T_G': camera_g_loss,
							  'acc': acc})


	return meter.get_val(), meter.get_str()
