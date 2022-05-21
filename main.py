'''
This file provides
train, test, visualization operations on market and duke dataset
'''


import argparse
import os
import ast
from core import ReIDLoaders, Base, train_an_epoch, test, plot_prerecall_curve, visualize
from tools import make_dirs, Logger, os_walk, time_now


def main(config):

	# init loaders and base
	loaders = ReIDLoaders(config)
	base = Base(config)

	# make directions
	make_dirs(base.output_path)

	# init logger
	logger = Logger(os.path.join(config.output_path, 'log.txt'))
	logger(config)


	assert config.mode in ['train', 'test', 'visualize']
	if config.mode == 'train':  # train mode

		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			start_train_epoch = base.resume_last_model()
		else:
			start_train_epoch = 0
		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs):
			# save model
			base.save_model(current_epoch)
			# train
			_, results = train_an_epoch(config, base, loaders, current_epoch)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))
			# if(current_epoch<(config.start_epoch+23)):
			# 	if ((current_epoch+1)%40==0):
			# 		mAP, CMC, _, _, _ = test(config, base, loaders)
			# 		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP,
			# 		                                                                 CMC))
			# else:
			# 	if ((current_epoch+1)%1==0):
			# 		mAP, CMC, _, _, _ = test(config, base, loaders)
			# 		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP,
			# 		                                                                 CMC))


		# test
		base.save_model(config.total_train_epochs)
		mAP, CMC,_,_,_ = test(config, base, loaders)
		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP, CMC))


	elif config.mode == 'test':	# test mode
		base.resume_from_model(config.resume_test_model)
		mAP, CMC, pres, recalls, thresholds = test(config, base, loaders)
		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP, CMC))
		logger('Time: {}; Test Dataset: {}, \nprecision: {} \nrecall: {}\nthresholds: {}'.format(
			time_now(), config.test_dataset, mAP, CMC, pres, recalls, thresholds))
		plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, 'none')


	elif config.mode == 'visualize': # visualization mode
		base.resume_from_model(config.resume_visualize_model)
		visualize(config, base, loaders)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--MultipleLoss', type=ast.literal_eval, default=False)
	parser.add_argument('--baseline', type=ast.literal_eval, default=False)
	parser.add_argument('--output_path', type=str, default='last/', help='path to save related informations')

	# dataset configuration  file:///media/l/storage/reid
	# parser.add_argument('--market_path', type=str, default='/home/lhf/ls/reid_data/market1501/')
	# parser.add_argument('--duke_path', type=str, default='/home/lhf/ls/reid_data/dukemtmc-reid/DukeMTMC-reID/')
	parser.add_argument('--market_path', type=str, default='/media/l/storage/reid/market1501/')
	parser.add_argument('--duke_path', type=str, default='/media/l/storage/reid/dukemtmc-reid/DukeMTMC-reID/')
	parser.add_argument('--msmt_path', type=str, default='/data/datasets/MSMT17_V1/')
	parser.add_argument('--njust_path', type=str, default='/data/datasets/njust365/')
	parser.add_argument('--wildtrack_path', type=str, default='/data/datasets/Wildtrack_crop_dataset/crop/')
	parser.add_argument('--combine_all', type=ast.literal_eval, default=False, help='train+query+gallery as train')
	#-----------------------------------------
	parser.add_argument('--train_dataset', nargs='+', type=str, default=['duke_market'], help='duke_market/market_duke/market/duke/msmt/njust_win/njust_spr/njust_both. support multi-dataset such as [market, duke]')
	parser.add_argument('--test_dataset', type=str, default='market', help='market, duke, msmt, njust_win/njust_spr/njust_both, wildtrack')
	parser.add_argument('--s_dataset', type=str, default='duke',
						help='market, duke, msmt, njust_win/njust_spr/njust_both, wildtrack')
	parser.add_argument('--t_dataset', type=str, default='market',help='market, duke, msmt, njust_win/njust_spr/njust_both, wildtrack')
	parser.add_argument('--s_camera_num', type=int, default=8, help='person count in a batch')
	parser.add_argument('--t_camera_num', type=int, default=6, help='images count of a person in a batch')
	# -----------------------------------------
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=4, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# data augmentation
	parser.add_argument('--use_rea', type=ast.literal_eval, default=False)
	parser.add_argument('--use_colorjitor', type=ast.literal_eval, default=True)
	# model configuration
	parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
	parser.add_argument('--pid_num', type=int, default=702, help='market:751(combineall-1503), duke:702(1812), msmt:1041(3060), njust:spr3869(5086),win,both(7729)')
	parser.add_argument('--margin', type=float, default=0.3, help='mar`gin for the triplet loss with batch hard')

	# train configuration
	parser.add_argument('--steps', type=int, default=1033, help='1 epoch include many steps')
	parser.add_argument('--steps_domain', type=int, default=800, help='1 epoch include many steps')
	parser.add_argument('--start_epoch', type=int, default=70, help='1 epoch include many steps')

	parser.add_argument('--milestones', nargs='+', type=int, default=[20,70], help='milestones for the learning rate decay')
	parser.add_argument('--m_2k_milestones', nargs='+', type=int, default=[20,70],help='milestones for the learning rate decay')
	parser.add_argument('--d_2k_milestones', nargs='+', type=int, default=[20,70],help='milestones for the learning rate decay')
	parser.add_argument('--D_domainn_milestones', nargs='+', type=int, default=[20,70],help='milestones for the learning rate decay')

	parser.add_argument('--base_learning_rate', type=float, default=0.0002)
	parser.add_argument('--m_2k_learn_rate', type=float, default=0.00012)
	# parser.add_argument('--d_2k_learn_rate', type=float, default=0.0003)
	parser.add_argument('--d_domain_learn_rate', type=float, default=0.0003)

	parser.add_argument('--m', type=float, default=1, help='m')
	parser.add_argument('--h', type=float, default=1, help='h')

	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--total_train_epochs', type=int, default=104)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')

	# test configurati
	parser.add_argument('--resume_test_model', type=str, default='model/model_89.pkl', help='')
	parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')

	# visualization configuration
	parser.add_argument('--resume_visualize_model', type=str, default='/path/to/pretrained/model.pkl',
						help='only availiable under visualize model')
	parser.add_argument('--visualize_dataset', type=str, default='',
						help='market, duke, only  only availiable under visualize model')
	parser.add_argument('--visualize_mode', type=str, default='inter-camera',
						help='inter-camera, intra-camera, all, only availiable under visualize model')
	parser.add_argument('--visualize_mode_onlyshow', type=str, default='pos', help='pos, neg, none')
	parser.add_argument('--visualize_output_path', type=str, default='results/visualization/',
						help='path to save visualization results, only availiable under visualize model')


	# main
	config = parser.parse_args()
	main(config)



