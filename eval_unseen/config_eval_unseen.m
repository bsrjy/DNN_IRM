% general configurations

% feature tpye
feat_line = 'AmsRastaplpMfccGf'

% unseen noise type
noise_line = 'white'

% number of times that each clean utterance is mixed with noise for training.
repeat_time = 1; 

test_list = ['config' filesep 'list120_short.txt'];

% cut noise into two parts, the first part is for training and the second part is for test
noise_cut = 0.5;

% create mixtures at certain SNR
mix_db = [-10, -5]; 

% use ideal ratio mask or ideal binary mask as learning target
is_ratio_mask = 1;

% 1. generate mixtures or not. 0: no, 1: yes.
is_gen_mix = 0;

% 2. generate features/masks or not. 0: no, 1: yes.
is_gen_feat = 0;

% 3. perform dnn test or not. 0: no, 1: yes.
is_dnn = 1;

model_path = 'D:\coding\Matlab\DNN_toolbox\DATA\white_factory_pink_babble\dnn\STORE\db-10  -5   0   5\model\ratio_white_factory_pink_babble_db-10  -5   0   5_AmsRastaplpMfccGf.mat'

TRAIN_STORE = 'D:\coding\Matlab\DNN_toolbox\DATA\white_factory_pink_babble\tmpdir\db-10  -5   0   5\feat\train_white_factory_pink_babble_AmsRastaplpMfccGf.mat';
