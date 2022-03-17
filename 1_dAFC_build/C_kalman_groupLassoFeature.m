clear;
clc;

%label_name = 'NC_EMCI';
%label_name = 'EMCI_LMCI';
label_name = 'NC_EMCI_LMCI';
for lamda = (0.01)
    for kf = (0.1:0.1:0.9)
        %  for kf = (0.9) %'NC_EMCI_LMCI'
        % for kf = (0.4) %'NC_EMCI'
        %for kf = (0.6) %'EMCI_LMCI'
        Folder_Original_Data = strcat('../0_A_data/',label_name);
        Kalmanfile = strcat('kalmancorr_',num2str(lamda),'_',num2str(kf),'.mat');
        load(fullfile(Folder_Original_Data,'/kalmancorr/',Kalmanfile));
        NUM = size(corr,1);
        frame = 130;
        adap_net=zeros(90,90,NUM,frame);
        for i=1:NUM
            for j=1:90
                for k=1:90
                    adap_net(j,k,i,:)=squeeze(corr{i,1}(j,k,:));
                end
            end
        end
        %    temp_net = reshape(adap_net,8100,NUM*frame);
        %   temp_net(all(temp_net==0,2),:)=[];   % 提取动态功能连接特征，去除行全为0的行 数据太多时不能用这个语句，会报错内存不足
        %动态功能连接特征
        k=0;
        len = zeros(i,1);
        temp_net = reshape(adap_net,8100,NUM*frame);
        for i=1:8100
            len(i) = length(find(temp_net(i,:)~=0));
            if length(find(temp_net(i,:)~=0)) >(NUM*frame/2)
                k=k+1;
            end
        end
        feature_net = zeros(k,NUM*frame);
        k=0;
        for i=1:8100
            if length(find(temp_net(i,:)~=0)) >(NUM*frame/2)
                k=k+1;
                feature_net(k,:) = temp_net(i,:);
            end
        end
        
        datas = cell(NUM,4);
        feature_net = permute(reshape(feature_net,k,NUM,frame),[2,3,1]);
        for i=1:NUM
            datas{i,1}=squeeze(feature_net(i,:,:));
            datas{i,2}=corr{i,2}; % person
            datas{i,3}=corr{i,3}; % date
            datas{i,4}=corr{i,4}; % label
        end
        disp(kf);
        file_path_name = strcat(Folder_Original_Data,'/GLfeatures/');
        if ~exist(file_path_name,'dir')   %该文件夹不存在，则直接创建
            mkdir(file_path_name);
        end
        Featurefile = strcat('kalmancorr_',num2str(lamda),'_',num2str(kf),'_',num2str(k),'.mat');   % 注意修改后缀名_one2one.mat
        save([file_path_name,Featurefile],'datas');
    end
end
