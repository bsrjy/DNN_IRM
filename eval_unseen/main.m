format compact
warning off;

config_eval_unseen

% create folders for storing data
tmp_path = ['DATA_UNSEEN' filesep];
if ~exist(tmp_path,'dir'); mkdir(tmp_path); end;
tmp_path = ['DATA_UNSEEN' filesep noise_line];
if ~exist(tmp_path,'dir'); mkdir(tmp_path); end;
tmp_path = ['DATA_UNSEEN' filesep noise_line filesep 'dnn'];
if ~exist(tmp_path,'dir'); mkdir(tmp_path); end;
tmp_path = ['DATA_UNSEEN' filesep noise_line filesep 'tmpdir'];
if ~exist(tmp_path,'dir'); mkdir(tmp_path); end;
TMP_DIR_STR = [pwd filesep 'DATA_UNSEEN' filesep noise_line filesep 'tmpdir'];

copyfile('config',['DATA_UNSEEN' filesep noise_line filesep 'config']);

% get # of test mixtures
num_mix_per_test_part = numel(textread(test_list,'%1c%*[^\n]'));



% generate mixture
if is_gen_mix == 1
	cd(['DATA_UNSEEN' filesep noise_line]);
	addpath(['..' filesep '..' filesep 'gen_mixture']);
	fprintf('\n----------------------------------------------------------\n');
	fprintf('generate mixtures\n');
	get_all_noise_test(noise_line, noise_cut, mix_db, test_list, TMP_DIR_STR);
	cd ..
	cd ..
end


% get features
if is_gen_feat == 1
	warning off;
	cd(['DATA_UNSEEN' filesep noise_line]);
	addpath(genpath([ '..' filesep '..' filesep '..' filesep 'get_feat']));
	%addpath('utility')
	%addpath('features')
	fprintf('\n----------------------------------------------------------\n');
	fprintf('generate features\n');
	total(feat_line, noise_line, -1, 1, num_mix_per_test_part, mix_db, is_ratio_mask, TMP_DIR_STR);
	cd ..
	cd ..
end


% mean & variance normalization for the features
if is_dnn == 1
	fprintf('\n----------------------------------------------------------\n');
	fprintf('test unseen noise\n');
	addpath(genpath('dnn_unseen'));
	cd(['DATA_UNSEEN' filesep noise_line]);
    % mean variance normalization
    mvn_store(noise_line, feat_line, mix_db, TMP_DIR_STR, TRAIN_STORE, num_mix_per_test_part);
    % dnn training/test
    run_every(noise_line, feat_line, mix_db, model_path, num_mix_per_test_part);
    cd ..
end
% get output from dnn
