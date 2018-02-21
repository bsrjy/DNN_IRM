function mvn_store(noise, feat, db, TEST_STORE, TRAIN_STORE, num_mix_per_test_part)
format compact;

fprintf(1,'MVNing Feat=%s Noise=%s\n', feat, noise);

tmp_str = strsplit(noise, '_');

% add support for multiple noise
noise_num = length(tmp_str);

% add support for multiple SNR
snr_num = length(db);

tmp_small_mix_cell = cell(1, num_mix_per_test_part * snr_num * noise_num);
tmp_small_noise_cell = cell(1, num_mix_per_test_part * snr_num * noise_num);
tmp_small_speech_cell = cell(1, num_mix_per_test_part * snr_num * noise_num);

% run through all snr and noise types
for i=1:snr_num
	cur_db = db(i);
	for j=1:noise_num
			cur_noise = tmp_str{j};
			root_path = [TEST_STORE filesep 'db' num2str(cur_db) filesep]
			speech_data_path = [root_path 'mix' filesep 'test_' cur_noise '_mix_aft2.mat'];
			load(speech_data_path);
			cur_cell = 1+num_mix_per_test_part*((i-1)*noise_num+j-1);
			tmp_small_speech_cell(cur_cell:cur_cell-1+num_mix_per_test_part) = small_speech_cell; 
			tmp_small_mix_cell(cur_cell:cur_cell-1+num_mix_per_test_part) = small_mix_cell;
			tmp_small_noise_cell(cur_cell:cur_cell-1+num_mix_per_test_part) = small_noise_cell;
	end
end

small_speech_cell = tmp_small_speech_cell;
small_noise_cell = tmp_small_noise_cell;
small_mix_cell = tmp_small_mix_cell;

% test_set
root_path = [TEST_STORE filesep 'db' num2str(db) filesep]
load([root_path 'feat' filesep 'test_' noise '_' feat '.mat']); %test set
test_data = feat_data; test_label = feat_label;
clear feat_data feat_label

% train_set
%train_path = [TRAIN_STORE 'feat' filesep 'train_' noise '_' feat '.mat']
train_path = TRAIN_STORE;

train_data = []; train_target = [];
disp(['loading ' train_path]);
load(train_path);
train_data = [train_data ; feat_data];
train_target = [train_target ; feat_label];
clear feat_data feat_label;

cv_portion = floor(0.1 * size(train_data, 1));

train_data(1:cv_portion,:) = [];
train_target(1:cv_portion,:) = [];

[a3, b3] = size(test_data);
fprintf(1,'test=%d x %d\n',a3,b3);

[train_data,para.tr_mu,para.tr_std] = mean_var_norm(train_data);
clear train_data train_target;
test_data = mean_var_norm_testing(test_data, para.tr_mu,para.tr_std);

save_mvn_prefix_path = ['MVN_STORE' filesep];
if ~exist(save_mvn_prefix_path,'dir'); mkdir(save_mvn_prefix_path); end;
MVN_DATA_PATH = [save_mvn_prefix_path 'allmvntrain_' noise '_' feat '_' num2str(db) '.mat']
save(MVN_DATA_PATH, 'para','test_data','test_label', 'DFI',...
 'small_mix_cell', 'small_noise_cell', 'small_speech_cell', 'c_mat', '-v7.3');%also saved test mixes

pause(2);

end
