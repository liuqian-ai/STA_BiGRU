function  [nets] = net_built_group_lasso(fmri,lamda)
num= size(fmri,1); % 元素总数
frame = size(fmri,2); % 脑区变化130
n=90; % 90
nets=zeros(n,n,num);
opts = [];
opts.q = 2;
opts.init = 2;
opts.tFlag = 5;
opts.maxIter = 1000;
opts.nFlag = 0;
opts.rFlag = 1;
opts.ind = [];
for i = 1:n
    Y=[];
    X=[];
    for j= 1 : num
        temp0 = reshape(fmri(j,:,1:90),frame,n);
        temp = temp0;
        Y = [Y ; temp(:,i)];
        temp(:,i)=0*temp(:,i);
        X = [X ;temp];
    end
    [MM,~]=size(Y);
    opts.ind =0:MM/num:MM;
    [x1, ~, ~] = mtLeastR(X ,Y,lamda,opts);
    nets(:,i,:)=x1;
    disp('***************************************************************');
    disp(['       ',num2str(100*i/n),'% of net construction is finished']);%, lamda_best is: ',num2str(lamda_best)]);
    disp('***************************************************************');
end


