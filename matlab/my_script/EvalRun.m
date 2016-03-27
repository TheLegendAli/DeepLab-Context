SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VOC_root_folder = '{DATA_ROOT}';

if has_postprocess == 2
   range_W=[5, 10]
   range_XY_STD=[40, 50, 60, 70, 80, 90, 100]
   range_RGB_STD=[3, 4, 5, 6, 7, 8, 9, 10]


   for w=1:length(range_W)
		bi_w=w;
		for x=1:length(range_XY_STD)
			bi_x_std=x;
			for r=1:length(range_RGB_STD)
				bi_r_std = r;
			%you dont need a folder for each variable because multiple values change together


				if learn_crf
    				post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch); 
  				else
    				post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
  				end
  				EvalSegResults(post_folder, feature_name, model_name, testset, feature_type, dataset, id, trainset, is_mat, is_argmax, has_postprocess);

			end
		end
	end


elseif has_postprocess == 1
	% initial or default values for crf
	bi_w           = 5; 
	bi_x_std       = 50;
	bi_r_std       = 3;

	pos_w          = 3;
	pos_x_std      = 3;
  if learn_crf
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch); 
  else
    post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
  end
  EvalSegResults;
else
  post_folder = 'post_none';
end
