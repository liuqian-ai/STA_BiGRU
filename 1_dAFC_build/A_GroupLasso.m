clc;
clear;

tic  % 计时开始       
%label_name = 'NC_EMCI';
%label_name = 'EMCI_LMCI';
label_name = 'NC_EMCI_LMCI';                                                                                                                                      

Signalfile = strcat('ROISignals_',label_name,'_total.mat');
Folder_Original_Data = strcat('../0_A_data/',label_name);
load (fullfile(Folder_Original_Data,Signalfile));
fmri = datas;
Folder = strcat('../0_A_data/',label_name,'/groupLasso');
%% 5 group Lasso
for lamda = (0.01)
     nets = net_built_group_lasso(fmri,lamda);
     save(fullfile(Folder,['nets_group_lasso_',num2str(lamda),'.mat']),'nets','labels');
end
toc  % 计时结束


     
 
