clc;
clear;
warning('off');
tic
% Ctrl+R ע��  Ctrl+T ȡ��ע��
%label_name = 'NC_EMCI';
%label_name = 'EMCI_LMCI';
label_name = 'NC_EMCI_LMCI';

for lamda = (0.01)
    %  for kf = (0.9) %'NC_EMCI_LMCI'
    % for kf = (0.4) %'NC_EMCI'
    % for kf = (0.6) %'EMCI_LMCI'
    for kf = (0.1:0.1:0.9)
        Signalfile = strcat('ROISignals_',label_name,'_total.mat');
        Lassofile = strcat('nets_group_lasso_',num2str(lamda),'.mat');
        Folder_Original_Data = strcat('../0_A_data/',label_name);
        load (fullfile(Folder_Original_Data,Signalfile));   %����ԭ�ļ�
        load (fullfile(Folder_Original_Data,'/groupLasso',Lassofile));  % ����GroupLasso�ļ�
        roi_data = datas;
        [M,F,R] = size(roi_data);  % M=������
        
        N = 4; % data person date label
        corr = cell(M,N);
        
        for k=1:M
            relation = zeros(90,90,130);
            for i=1:90
                for j=1:90
                    if abs(nets(i,j,k))==0
                        continue;
                    else
                        temp =  reshape(roi_data(k,:,1:90),130,90);
                        % ���� ��һ������130��ʱ���
                        input = temp(:,i);
                        % ��� ��һ������130��ʱ���
                        output = temp(:,j);
                        z = [output input];
                        nn = [0 1 1];
                        [thm,yhat,P,phi] = rarx(z,nn,'kf',kf);
                        relation(i,j,:)=thm;
                    end
                end
            end
            corr{k,1}=relation;
            corr{k,2}=persons(k,:);
            corr{k,3}=dates{k,1};
            corr{k,4}=labels(k,:);
        end
        disp(kf);
        Kalmanfile = strcat('kalmancorr_',num2str(lamda),'_',num2str(kf),'.mat');
        save([Folder_Original_Data,'/kalmancorr/',Kalmanfile],'corr');
    end
end
toc


