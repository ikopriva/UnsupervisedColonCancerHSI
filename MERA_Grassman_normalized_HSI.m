%==========================================================================
%
% MERA_Grassmann_normalized_HSI
%
% (c) Ivica Kopriva January 2026
%
%==========================================================================

clear all; close all; clc

addpath(genpath('.'));

%% load dataset
% It is assumed that hyperspectral data are located in
% the local directory. They are available for download from the link below:
% https:\\data.fulir.irb.hr\islandora\object\irb:538

% It is further assumed that hyperspectral images with normlized spectra 
% accoring to Algorithm 1 (equations 12 to 18) are located in the local
% directory. For this purpose use the available Pythone code 

mean_angle_target = 3.4;
angle_normalized_vs_not_normalized = 1.5; % when to switch from not normalized to normalized HSI

p = 5;              % patch size: (2p+1) x (2p+1)
% p=0 ===> no patch

% rx_1=25; rx_2=25;  6x4 patches
rx_1=35; rx_2=36;  % 3x4 patches
Num_labels = rx_1*rx_2; % number of labels per category used to construct Grassmann points

Nclass = 2;

dimSubspace = 50;   % subspace dimensions of Grassmannian points

numwav=351;          % 450nm to 800 nm;  

blck_tp=0; blck_tn=0; blck_fp=0; blck_fn=0;  % for estimation of micro performance

% load ground truth data
load GT_train
load GT_test

GT = GT_train;
GT(:,:,18:27)=GT_test;
clear GT_train GT_test

h_dataset = waitbar(0,'Progressing SSL-MPRI classification on dataset level. Please wait...');

for itr = 1:27
     waitbar(itr/27,h_dataset)

    tstart = tic; % estimate CPU time per image

    blck_tp_img=0; blck_tn_img=0; blck_fp_img=0; blck_fn_img=0;

    % "Normal HSI"
    if itr < 18
        filename=strcat('HSI_train_',num2str(itr),'.h5');
    else
        filename=strcat('HSI_test_',num2str(itr),'.h5');
    end
    img=h5read(filename,'/img');

    blck_count = 0;

    tstart_train=tic; 

    % HSIs with normalized spectra
    if itr < 18
        filename=strcat('HSI_train_',num2str(itr),'_normalized,'.h5');
    else
        filename=strcat('HSI_test_',num2str(itr),'_normalized','.h5');
    end
    img_normalized = h5read(filename,'/img');

    [H W B] = size(img)

    X = reshape(shiftdim(img,2),B,W*H);

    Input_gt = GT(:,:,itr); % "1" is because classes are 1 and 2 (noncancer and cancer)

    % define block size
    %dh=230; dw=258;
    dh=346; dw=345;
    HI=floor(H/dh)*dh; WI=floor(W/dw)*dw;

    i_h = 0;
    for hh= 1:dh:HI
        i_h = i_h +1;
        hh_s = hh;
        if hh_s <= HI-dh
            hh_e = hh_s + (dh-1);
        else
            hh_e = HI;
        end

        i_w=0;
        for ww=1:dw:WI
            i_w = i_w + 1;
            ww_s = ww;
            if ww <= WI-dw
                ww_e = ww_s + (dw-1);
            else
                ww_e = WI;
            end

            blck_count = blck_count + 1;

            %select dx by dw region
            patch_img = double(img(hh_s:hh_e,ww_s:ww_e,:));
            patch_labels = double(Input_gt(hh_s:hh_e,ww_s:ww_e));
            nc = sum(sum(patch_labels)); nnc=sum(sum(not(patch_labels)));

            %%
            if (length(unique(patch_labels)) == Nclass) && (Num_labels <= nc) && (Num_labels <= nnc)   % skip blocks with less than Nclass labels
                TrnLabels=[];
                TestLabels=[];
                Tr_idx_C=[];
                Te_idx_C=[];
                Te_idx_R=[];
                Tr_idx_R=[];

                display('Training image:')
                itr
                display('Block row start:')
                hh_s
                display('Block column start:')
                ww_s

                %% Construct representatives Y0 and Y1
                [nx nw B] = size(patch_img);

                XX = reshape(shiftdim(patch_img,2),size(patch_img,3),nx*nw);

                % Select randomly Num_labels location
                Num=Num_labels;
                ij=0;
                for i=1:nx
                    for j=1:nw
                        ij=ij+1;
                        All_idx_R(ij,1)=i;
                        All_idx_C(ij,1)=j;
                    end
                end
                ind_all = sub2ind([nx nw], All_idx_R, All_idx_C);
                ind_all_perm = randperm(nx*nw);
                ind_train = ind_all_perm(1:Num);
                [Tr_idx_R Tr_idx_C] = ind2sub([nx nw], ind_train);
                for i=1:numel(Tr_idx_R)
                    TrnLabels(1,i)=patch_labels(Tr_idx_R(i),Tr_idx_C(i));
                end

                ind_test=setdiff(ind_all_perm,ind_train);
                [Te_idx_R Te_idx_C] = ind2sub([nx nw], ind_test);
                for i=1:numel(Te_idx_R)
                    TestLabels(1,i)=patch_labels(Te_idx_R(i),Te_idx_C(i));
                end

                vec_train_patch = XX(:,ind_train);

                ind_0 = (TrnLabels == 0);
                noncancer_mean = mean(vec_train_patch(:,ind_0),2);
                ind_1 = (TrnLabels == 1);
                cancer_mean = mean(vec_train_patch(:,ind_1),2);
                mean_angle = acos((noncancer_mean'*cancer_mean)/norm(noncancer_mean)/norm(cancer_mean))*180/pi

                counter = 0; old_angle = mean_angle;
                Tr_idx_R_new = Tr_idx_R; Tr_idx_C_new = Tr_idx_C; TrnLabels_new = TrnLabels;
                ind_all_perm_new = ind_all_perm; ind_train_new = ind_train;

                while mean_angle < mean_angle_target && counter < 100
                    counter = counter + 1;
                    ind_all_perm = randperm(nx*nw);
                    ind_train = ind_all_perm(1:Num);
                    [Tr_idx_R Tr_idx_C] = ind2sub([nx nw], ind_train);
                    for i=1:numel(Tr_idx_R)
                        TrnLabels(1,i)=patch_labels(Tr_idx_R(i),Tr_idx_C(i));
                    end

                    ind_test=setdiff(ind_all_perm,ind_train);
                    [Te_idx_R Te_idx_C] = ind2sub([nx nw], ind_test);
                    for i=1:numel(Te_idx_R)
                        TestLabels(1,i)=patch_labels(Te_idx_R(i),Te_idx_C(i));
                    end

                    for n=1:Num
                        vec_train_patch(:,n)= patch_img(Tr_idx_R(n), Tr_idx_C(n),:);
                    end

                    ind_0 = (TrnLabels == 0);
                    noncancer_mean = mean(vec_train_patch(:,ind_0),2);
                    ind_1 = (TrnLabels == 1);
                    cancer_mean = mean(vec_train_patch(:,ind_1),2);
                    mean_angle = acos((noncancer_mean'*cancer_mean)/norm(noncancer_mean)/norm(cancer_mean))*180/pi;
                    if old_angle < mean_angle
                        old_angle = mean_angle;
                        ind_test=setdiff(ind_all_perm,ind_train);
                        [Te_idx_R Te_idx_C] = ind2sub([nx nw], ind_test);
                        for i=1:numel(Te_idx_R)
                            TestLabels(1,i)=patch_labels(Te_idx_R(i),Te_idx_C(i));
                        end
                    end
                end

                old_angle
                mean_angle
                blck_angle(i_h,i_w,itr) = old_angle
                norm_flag(i_h,i_w,itr) = 0;

                % parameters for MERA network
                R_opt=5;
                lambda_opt = 1e-12;

                if old_angle < angle_normalized_vs_not_normalized % Use data from spectrum normalized HSIs
                    norm_flag(i_h,i_w,itr) = 1;
                    patch_img = double(img_normalized(hh_s:hh_e,ww_s:ww_e,:));
                    for n=1:Num
                        vec_train_patch(:,n)= patch_img(Tr_idx_R(n), Tr_idx_C(n),:);
                    end
                    % parameters for MERA network
                    R_opt=2;
                    lambda_opt = 1e-9;
                end

                %% STEP 2: MERA
                XMV{1} = vec_train_patch(1:70,:);
                XMV{2} = vec_train_patch(71:140,:);
                XMV{3} = vec_train_patch(141:210,:);
                XMV{4} = vec_train_patch(211:280,:);
                XMV{5} = vec_train_patch(281:350,:);

                nV=length(XMV);

                rX= [rx_1 rx_2 rx_1 rx_2 nV]; % Assuming: N=Num=rx_1*rx_2;
                %
                paras_mera.rX = rX;
                paras_mera.R{1}=[R_opt,R_opt];
                paras_mera.lambda = lambda_opt;

                % MERA SC
               [S,mera]=MERA_MSC(XMV,paras_mera);
               

                ACC_1  =  Accuracy(labels_est,TrnLabels+1) ;
                ACC_2  = Accuracy(double(not(labels_est-1))+1,TrnLabels+1) ;
                if ACC_1 < ACC_2
                    labels_est = double(not(labels_est-1))+1;
                    display('*****************************')
                    display('Label permutation corrected !!!!!!!!!!')
                    display('*****************************')
                end

                ACC_blck(i_h,i_w,itr)  =  Accuracy(labels_est,TrnLabels+1)


                % Noncancer class
                LL = find(labels_est == 1);
                for i=1:length(LL)
                    blk = patch_extractor(patch_img,Tr_idx_R(LL(i)),Tr_idx_C(LL(i)),p);
                    X0(:,i) = squeeze(mean(mean(blk,1),2));
                end

                % Cancer class
                LL = find(labels_est == 2);
                for i=1:length(LL)
                    blk = patch_extractor(patch_img,Tr_idx_R(LL(i)),Tr_idx_C(LL(i)),p);
                    X1(:,i) = squeeze(mean(mean(blk,1),2));
                end

                n0=size(X0,2)
                n1=size(X1,2)
                cancer_vs_noncancer(i_h,i_w,itr)=n1/n0

                    % Estimate orthonormal bases: Grassmann points
                    XX = [X0 X1];
                    labels = [ones(1,size(X0,2)) 2*ones(1,size(X1,2))];
                    [affinity_x, B_x, begB_x, enddB_x, mu_X] = average_affinity(XX,labels,dimSubspace);

                    % Testing
                    for i=1:numel(Te_idx_R)
                        if mean_flag == 0
                            X_out(:,i)=patch_img(Te_idx_R(i),Te_idx_C(i),:);
                        elseif mean_flag == 1  % assign mean value
                            blk = patch_extractor(patch_img,Te_idx_R(i),Te_idx_C(i),p);
                            X_out(:,i) = squeeze(mean(mean(blk,1),2));
                        end
                    end

                    for el=1:2
                        X_outm = X_out - mu_X(:,el);    % make data zero mean for distance calculation
                        BB=B_x(:,begB_x(el):enddB_x(el));
                        Xproj = (BB*BB')*X_outm;
                        Dproj = X_outm - Xproj;
                        D(el,:) = sqrt(sum(Dproj.^2,1));
                    end
                    [~, testLabels_est] = min(D);
                    testLabels_est = testLabels_est - 1;
                    clear D X_out X0 X1
            else  % assign the same label to all pixels in the block
                display('ONE LABEL ONLY !!!!!!!!!!')

                TrnLabels=[];
                TestLabels=[];
                Tr_idx_C=[];
                Te_idx_C=[];
                Te_idx_R=[];
                Tr_idx_R=[];

                display('Training image:')
                itr
                display('Block row start:')
                hh_s
                display('Block column start:')
                ww_s

                i = min(min(patch_labels));
                [R C]=find(patch_labels==i);
                Num=Num_labels;
                idx_rand=randperm(numel(C));

                Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
                Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
                Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
                Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
                TrnLabels=[TrnLabels ones(1,Num)*i];
                TestLabels=[TestLabels ones(1,numel(C)-Num)*i];
                testLabels_est(1,:)=TestLabels;
                labels_est = TrnLabels + 1;
            end

            gtest=testLabels_est;
            gt = patch_labels;

            for i=1:numel(labels_est)
                gth(Tr_idx_R(i),Tr_idx_C(i))=labels_est(i)-1;
            end

            for i=1:numel(Te_idx_R)
                gth(Te_idx_R(i),Te_idx_C(i))=gtest(i);
            end

            GTh_img(hh_s:hh_e,ww_s:ww_e,itr) = gth;
            GT_img(hh_s:hh_e,ww_s:ww_e) = gt;

            TP = sum(double(and(logical(TestLabels),logical(testLabels_est))));
            TN = sum(double(~or(logical(TestLabels),logical(testLabels_est))));
            FN = sum(TestLabels) - TP;
            FP = sum(double(~logical(TestLabels))) - TN;

            % image based performance
            blck_tp_img = blck_tp_img + TP;
            blck_tn_img = blck_tn_img + TN;
            blck_fp_img = blck_fp_img + FP;
            blck_fn_img = blck_fn_img + FN;
            F1_train_img = 2*blck_tp_img/(2*blck_tp_img + blck_fp_img + blck_fn_img)

            % micro performance
            blck_tp = blck_tp + TP;
            blck_tn = blck_tn + TN;
            blck_fp = blck_fp + FP;
            blck_fn = blck_fn + FN;

            F1_train_micro = 2*blck_tp/(2*blck_tp + blck_fp + blck_fn)
            BACC_train_micro = (blck_tp/(blck_tp + blck_fn) + blck_tn/(blck_tn+blck_fp))/2

            training_time(itr) = toc(tstart_train);
           
            clear patch_imag patch_labels testLabels_est gth gt
        end
    end

     cmap=[0 0 1; 1 1 0];  %blue:: noncancer;  yellow: cancer;
     GT_int=uint8(GT_img(:,:));
     GTh_img_int = uint8(GTh_img(:,:,itr));
     GTh_rgb = ind2rgb(GTh_img_int,cmap);

    if itr < 18
        filename_gth=strcat('GTh_HSI_train_',num2str(itr),'_',num2str(Num_labels_per_class),'_MERA','.tiff') ;
    else
        filename_gth=strcat('GTh_HSI_test_',num2str(itr),'_',num2str(Num_labels_per_class),'_MERA','.tiff') ;
    end
    imwrite(GTh_rgb,filename_gth,'tiff')

    display('Image based performance metrics:')
    % image based performance
    img_sens(itr) = blck_tp_img/(blck_tp_img + blck_fn_img)
    img_spec(itr) = blck_tn_img/(blck_tn_img + blck_fp_img)
    img_bacc(itr) = (img_sens(itr) + img_spec(itr))/2
    img_F1(itr) = 2*blck_tp_img/(2*blck_tp_img + blck_fp_img + blck_fn_img)
    img_IoU(itr) = img_F1(itr)/(2-img_F1(itr))
    img_ppv(itr) = blck_tp_img/(blck_tp_img + blck_fp_img)

    display('Micro performance metrics:')
    % micro performance
    micro_sens = blck_tp/(blck_tp + blck_fn)
    micro_spec = blck_tn/(blck_tn + blck_fp)
    micro_bacc = (micro_sens + micro_spec)/2
    micro_F1 = 2*blck_tp/(2*blck_tp + blck_fp + blck_fn)
    micro_IoU = micro_F1/(2-micro_F1)
    micro_ppv = blck_tp/(blck_tp + blck_fp)

    filename = strcat(' MERA_Grassmann_HSI_validation_results_',num2str(Num_labels_per_class),'_labels_per_class')
    save(filename, 'micro_sens', 'micro_spec', 'micro_bacc', 'micro_F1',...
        'micro_IoU', 'micro_ppv', 'itr', 'blck_tp', 'blck_tn', 'blck_fp', 'blck_fn',...
        'img_sens', 'img_spec', 'img_bacc', 'img_F1', 'img_IoU', 'img_ppv')

    cpu_time(itr) = toc(tstart)
end
close(h_dataset) % end-of-dataset-loop