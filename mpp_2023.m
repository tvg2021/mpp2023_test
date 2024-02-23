function [ output_args ] = mpp24_exp(mode)
pTag='MPP_2023'; 

if nargin == 0
    mode = 1;
end
    NSI = 1;        % Noise Supressing Index (1/NSI) дисперсия шума.
    NSIstr = '1';   % num2str(NSI);
    reg_mode = 1;   % 1-RidgeRegression, 2-StepWise Regression
    % 
    initW = 3;      % 2:Wnew=1/(N*(N+1));  (?)
                    % 3:Wnew=W(N-1) -
                    % 4:Wnew=W(N-1)+1/(N*(N+1)*log(N+1)))
                    
    H = 20;         % 
        
    fsBase  = 1;  % FixedShare Alpha (1 как в статье!)
    fsMode  = 1; % 1-Just plane FixedShare version, 2-use history
    fsTag   = ['FS' num2str(fsBase) 'M' num2str(fsMode)]; 
      
    if fsBase>0  
        fsAlpha  = 1/fsBase;       % First value of FixedShare Alpha 
    else
        fsAlpha  = 1/abs(fsBase);  % First value of FixedShare Alpha 
    end
    
    s_prefix = ...
        [pTag  'Iw' num2str(initW) '_' fsTag '-H' num2str(H)];
    
    %% Parameters
    noise_scale = 1/NSI;
    sp = 1;           % sIGMA pOWER ^2 ^4 ^6 
    sigma =1/(10^sp); % 
    M = 4;            % 
    T = 3000;         % 
    sCount = 9;       %   
    K = 20; %         %    
    d = 1;            %  
    b = 21;           % 
                      
    b_upper = b +0.000001; 
    etaV = 1/(2*b*b); % AA
    etaW = 1/(8*b*b); % WA
    
    %% CONSTANTS
    figname_suffix = s_prefix;                                   
    fig_position = [200 50 900 550];             
    
    Frol = nan(K,T);  % 
    Fcum = nan(K,T);  % 
    Fwrm = nan(K,T);  % ConvexSmoothing
    
    %%  
    WWv = nan(T,T);   % Vovk
    WWw = nan(T,T);   % Warmuth
    nWWv = nan(T,T);  % nATIVE  Vovk
    nWWw = nan(T,T);  % nATIVE  Warmuth
    IWM  = nan(T,T);  % Internal Weights Matrix 
    eList = 100:300:2700; 
    eList = [H eList];    
    
    %% Point Predictions
    Yhat  = nan(T,T); 
    YhatV = nan(T,T); 
    YhatW = nan(T,T); 
    
    %%  Poin errors
    Rts   = nan(T,T);  
    R1t   = nan(1,T);  
    Rct   = nan(1,T);  
    Rvvk  = nan(1,T);  
    Rwrm  = nan(1,T);  
    nRvvk = nan(1,T);  
    nRwrm = nan(1,T);  
    
    Loc_opt_exp = nan(K,sCount); 
        
    %% 1. 
    DataGenerating = [1];
    for ii = DataGenerating
        rng(1024); % To produce a predictable sequence of numbers rng(1023)
        GMW = randn(M,K); % Generating Random Model Weigts 
        % Generatite samples
        % Segments 
        [X,Y,Segments,bw,iShift,mIndex,b_trn] = ...
                    gen_samples(T,M,K,GMW,sCount,noise_scale,b_upper);
        
        % 
        rng(2047); 
        [Xtst,Ytst,Segments_tst,~,iShift_tst,mIndex_tst,b_tst] = ...
                   gen_samples(T,M,K,GMW,sCount,noise_scale,b_upper);
        %% 
        test_sum = norm(Segments - Segments_tst)+...
                   norm(mIndex   - mIndex_tst);
        if abs(test_sum)>0
            error('Illegal samples!!!');
        end
        tst2trn = 0;
        if tst2trn
            % If tst2trn, than change (Xtst, X) to (Ytst, Y)
            Xtmp = Xtst;
            Ytmp = Ytst;
            Xtst = X;
            Ytst = Y;
            X = Xtmp;
            Y = Ytmp;
        end    
        % 
        csvwrite('train_data.csv',[X Y]);
        csvwrite( 'test_data.csv',[Xtst Ytst]);
        q=1;
    end
    
    %% Local models 
    for t = H : T  
        % 
        Xt = X(t-H+1:t,:);  % {Xt Yt} 
        Yt = Y(t-H+1:t);
        Ft = get_reg_model(Yt,Xt,sigma,reg_mode); % 
        Frol(:,t) = Ft;                           % 
        
        % 
        % 
        for s = t+1 : T
            Xs = X(s,:);
            % 
            L2 = get_loss2(Xs,Ft,Y(s));
            Rts(t,s) = L2;
        end
        
        if t<T   
            Xs = X(t+1,:);
            yh = Xs*Ft;
            R1t(t+1) = (Y(t+1)-yh)^2;
            % 
            if ~(R1t(t+1)==Rts(t,t+1)) 
                error('Illegal R1t value!');
            end
        end    
        
        % 
        cFt = get_reg_model(Y(1:t),X(1:t,:),sigma,reg_mode);
        Fcum(:,t) = cFt;
        % 
        s = t+1;
        if s<T
            Xs = X(s,:);
            yh = Xs*cFt;
            dd = Y(s)-yh;
            Rct(1,s) = dd*dd; % 
        end
    end
    
    %% 
    C_const = 2.10974;
    w_series=zeros(1,T);
    for t = H+1:T 
        w_current = 1/(C_const*(t-H+1)*((log(t-H+1))^2));
        IWM(t,t) = w_current;  
        w_series(t) = w_current; 
    end
    www=cumsum(w_series);
    q=1;
        
    if initW == 3 
        nWWv(H:T,H) = 1;  
        nWWw(H:T,H) = 1;  
    elseif initW == 2
        for i = H:T
            nWWv(i,H) = w_series(i); 
            nWWw(i,H) = w_series(i); 
        end
        nWWv(H,H) = 1;
        nWWw(H,H) = 1;
        q=1;
    else 
        error('Illegal value of initW !');
    end
  
    q = 1;
    draw_model_pictures = []; 
    for dd = draw_model_pictures
        % 
        h = figure('Position',fig_position);
        xx = (1:T)';
        yH = 20;
    
        subplot(2,1,1);
        for i=1:sCount-1
            xL = Segments(i,2);    % xL_abel 
            plot([xL xL],[-yH yH]','-g','LineWidth',1);  hold on;
            plot([i*bw i*bw],[-yH yH]','-y','LineWidth',2);  hold on;
        end
        plot(xx,Y,'-b'); ylabel('Y'); xlabel('Time'); hold on;
        rct = [(iShift*bw-200) -(yH-2) 400 2*(yH-2)];
        xlim([1 T]); ylim([-20 20]);
        rectangle('Position',rct,'EdgeColor','r');
        xlim([1 T]); ylim([-yH yH]);
        title('Full TS (T=3000)');
    
        subplot(2,1,2);
        plot([iShift*bw iShift*bw],[-yH yH]','-g','LineWidth',2);  hold on;
        xx = (iShift*bw-200:iShift*bw+200);
        yy = Y(xx)';
        plot(xx,yy,'-r'); ylabel('Y'); xlabel('Time');
        xlim([xx(1) xx(end)]); ylim([-20 20]);
        title('Zoomed Fragment of TS');
            
        saveas(h,['Y+ZoomedFragment-' figname_suffix '.png']);
        delete(h);
        % Рисуем коэффициенты частных моделей на фоне истинных значений
        %% Можно развернуть в 3D по времени
        h = figure('Position',fig_position);
        plot(Frol); ylabel('Rolling Models'); xlabel('Numbers of regression coefficients');
        hold on
        plot(GMW');  plot(GMW','LineWidth',3);
        title('Generating (thick) and Rolling Experts (thin) models');
        xlim([1 20]);
        saveas(h,['CoeffsAndEstimations-' rcd(figname_suffix) '.png']);
        delete(h);
        
        h = figure('Position',fig_position);
        plot(Frol');
        title('Time Course of Rolling Regression Model Coefficients');
        xlabel('Time');
        ylabel('w');
        saveas(h,['CoeffTimeCourse-' rcd(figname_suffix) '.png']);
        delete(h);
        q=1;
    end
            
    %% 
    % 
    % 
    % 
    % 
    % 
    % 
    
    %% 
    for t = H:T-1
        n = t-H+1; % exp number
        if initW==2
              WWv(t,t+1) = 1/((n+1)*log(n+1)*log(n+1));  % from paper
              WWw(t,t+1) = 1/((n+1)*log(n+1)*log(n+1));  
              % 
              % 
        elseif initW==3      
              WWv(t,t+1)=1;  % 
              WWw(t,t+1)=1;  % 
        elseif initW==4
            WWv(t,t+1)=1/(n+1);  % 
            WWw(t,t+1)=1/(n+1);  % 
        else
            WWv(t,t+1)=1/n;
            WWw(t,t+1)=1/n;  
        end
    end             
    
    %% 
    %% 
    %% Main part of program. 
    %% 
    %% 
    
    for t = H+1:T    % 
        % 
        % 
        % 
        % 
        N = t-H; % 
        % 
        % 
        
        if N>1 % 
            %% 
            if initW==2 % 
                % 
                WWw(H:H+N-2,t) = WWw(H:H+N-2,t-1);  % 
                WWv(H:H+N-2,t) = WWv(H:H+N-2,t-1);  % 
            elseif initW==3
                WWw(H:H+N-2,t) = WWw(H:H+N-2,t-1); % 
                WWv(H:H+N-2,t) = WWv(H:H+N-2,t-1); % 
            elseif initW==4 
                WWw(H:H+N-2,t) = WWw(H:H+N-2,t-1); % 
                WWv(H:H+N-2,t) = WWv(H:H+N-2,t-1); % 
            else
                  error('Illegal value of initW!');  
            end
        end
        
        % 
        % 
        for n = 1 : N
            j = H-1+n;     % 
            WWw(j,t) = WWw(j,t)*exp(-etaW*Rts(j,t));
            WWv(j,t) = WWv(j,t)*exp(-etaV*Rts(j,t));
        end
        
        %% Строим агрегаторы и оцениваем их потери
        x_cur = X(t,:);     % Текущий входной вектор ("сигнал")
        y_cur = Y(t);       % Текущий отклик
        
        F_cur = Frol(:,H:H+N-1);   % F_cur - N active experts 
                
        %% WA
        w_cur_w = WWw(H:H+N-1,t);  % Weights experts (Warmuth)
        
        % 
        if fsMode == 1  % Standard Fixed Share for Warmuth aggregator 
            % Correct fsAlpha using number
            % N of active experts
            % Normalisation
            sumW = sum(w_cur_w);
            w_cur_w = w_cur_w/sumW;
        
            % 
            % 
            if fsBase>0
                fsAlpha = 1/(fsBase+N);
            else
                fsAlpha = 1/abs(fsBase);
            end
        
            % No need to normalization
            w_cur_w = fsAlpha/length(w_cur_w)+(1-fsAlpha)*w_cur_w;
        elseif fsMode==2  % Normalization needed !
            %% 
            qq = 1; 
            if N > 1  % 
                q=1;  % 
                       
                dd = 0;      
                for j = 1:N-1
                    dd = dd+WWw(H+j,t);
                end
                dd = dd/(N-1);
                w_cur_w(N) = w_cur_w(N)+dd;
                w_cur_w = w_cur_w/sum(w_cur_w);
            end          
            w_cur_w = w_cur_w/sum(w_cur_w);    %% 
        else
            error(['Illegal fsMode value - ' num2str(fsMode) ' !']);
        end

        % Warmuth
        yH_w = getGammaW(w_cur_w,F_cur,x_cur); % 
%       y_h =(x_cur*F_cur)*w_cur_w;  
        Fwrm(:,t) = F_cur*w_cur_w;  % Save to Fwrm  
        Rwrm(t) = (yH_w-y_cur)*(yH_w-y_cur);  % R2 Losses
        
        %% AA
        w_cur_v = WWv(H:H+N-1,t);  % Cumulated weights (Vovk)
        % 
        sumW = sum(w_cur_v);
        w_cur_v = w_cur_v/sumW;              
        
        %%  fsMode !!! 
        if fsMode==1     %% 
            w_cur_v = fsAlpha/length(w_cur_v)+(1-fsAlpha)*w_cur_v;
        elseif fsMode==2 %% 
            q=1;
            if N > 1  % 
                q=1;  % 
                      % 
                dd = 0;      
                for j = 1:N-1
                    dd = dd+WWv(H+j,t);
                end
                dd = dd/(N-1);
                w_cur_v(N) = w_cur_v(N)+dd;
            end          
            w_cur_v = w_cur_v/sum(w_cur_v);  
        else
            error('Illegal fsMode!');
        end    
                
        % 
        yH_v = getGammaV(w_cur_v,F_cur,x_cur,b,etaV); % 
        Rvvk(t) = (yH_v-y_cur)*(yH_v-y_cur);   % 
        
        %% Parallel calculation (nWWw, nWWv)
        %% 
        % 
        if N>0
            nWWw(H:T,t) = nWWw(H:T,t-1);  % W
            nWWv(H:T,t) = nWWv(H:T,t-1);  % V 
        end
                
        %% Exp correction of
        %% expert weights
        for n = 1 : N
            j = H-1+n;     %  n in WW and Rts matrix
            nWWw(j,t) = nWWw(j,t)*exp(-etaW*Rts(j,t));
            nWWv(j,t) = nWWv(j,t)*exp(-etaV*Rts(j,t));
        end
        
        %% 
        nw_cur_w = nWWw(H:H+N-1,t);  % Warmuth
        
        % 
        nsumW = sum(nw_cur_w);
        nw_cur_w = nw_cur_w/nsumW;
        if fsMode==1 
            nw_cur_w = fsAlpha/length(nw_cur_w)+(1-fsAlpha)*nw_cur_w;
            %%w_cur_w = fsAlpha/length(w_cur_w)+(1-fsAlpha)*w_cur_w;
        elseif fsMode==2
            % Repeat mode 1!
            nw_cur_w = fsAlpha/length(nw_cur_w)+(1-fsAlpha)*nw_cur_w;
        else
            error('Illegal fsMode!');
        end   
        %% nw_cur_w = fsAlpha/length(nw_cur_w)+(1-fsAlpha)*nw_cur_w;
        
        nyH_w = getGammaW(nw_cur_w,F_cur,x_cur); % 
        nRwrm(t) = (nyH_w-y_cur)*(nyH_w-y_cur);  % 
        nRwrm_cur = nRwrm(t);                    % 
        
        nw_cur_v = nWWv(H:H+N-1,t);  % 
        nsumW = sum(nw_cur_v);
        nw_cur_v = nw_cur_v/nsumW; 
        
        if fsMode==1 
            nw_cur_v = fsAlpha/length(nw_cur_v)+(1-fsAlpha)*nw_cur_v;
        elseif fsMode==2 
            %% !!! mode1 !!!
            nw_cur_v = fsAlpha/length(nw_cur_v)+(1-fsAlpha)*nw_cur_v;
        else
            error('Illegal fsMode!');
        end   
        
        nyH_v = getGammaV(nw_cur_v,F_cur,x_cur,b,etaV); % Получили прогноз 
        nRvvk(t) = (nyH_v-y_cur)*(nyH_v-y_cur);   % Вычисляем квадр. потери
        nRvvk_cur = nRvvk(t); % 
        
        %% Correct/ weights of sleeping experts
        for n = N+1:T-H
            j = H-1+n;    % 
            nWWw(j,t) = nWWw(j,t)*exp(-etaW*nRwrm_cur);
            nWWv(j,t) = nWWv(j,t)*exp(-etaV*nRvvk_cur);
        end
        q = 1;
    end 
    %% Of MPP Aggregation 
    %% View results
    q = 1;
    
%     h = figure('Position',fig_position);
%     figname = ['R1-WA-AA-Losses-' rcd(figname_suffix) '.png'];
%     legend_list =[];
    tt    = (H+1:T);
    Wcl   = cumsum(Rwrm(tt));    
    VcL   = cumsum(Rvvk(tt));    
    nWcl  = cumsum(nRwrm(tt));   
    nVcL  = cumsum(nRvvk(tt));   
    R1cL  = cumsum(R1t(tt));     
        
% %     plot(tt,R1cL,'g','LineWidth',1);      hold on  
% %     plot(tt,nWcl,'--b','LineWidth',1);    hold on
% %     plot(tt,nVcL,':m','LineWidth',1);     hold on
% %     plot(tt,Wcl,'-b','LineWidth',2);      hold on
% %     plot(tt,VcL,':r','LineWidth',2);      hold on
% %     
% %     title(rcd(['Cum. Losses of R1, nWA, nAA, WA and AA Aggregators ('... 
% %         figname_suffix ')'])); 
% %     %% R1cL,nWcl,nVcL,Wcl,VcL
% %     
% %     % End points
% %     for s = 1 : sCount
% %         xx = [Segments(s,2);Segments(s,2)]
% %         yy =  ylim';
% %         plot(xx,yy,':r','LineWidth',0.5);
% %     end
% %     %grid on
% %     legend([{'   R1'} {'  nWA'} {'  nAA'} {'   WA'} {'   AA'}],...
% %                                 'location','northwest');
% %     saveas(h,figname);
% %     delete(h);
    
    
    %% 
    Segm_count = size(Segments,1); % Число сегментов
    BLM_list = nan(Segm_count,K);  % Best hindsight model for each segment
    %% Best local
    for s = 1:Segm_count 
        % - 
        e_local_ind = (Segments(s,1)+H-1:Segments(s,2)); % Валидные модели       
        % 
        tt = (Segments(s,1):Segments(s,2));
        %% 
        [BLM,BLM_ind] = get_best_local_expert(X,Y,tt,Frol,e_local_ind)
        BLM_list(s,:) = BLM';
       
        %% Sorting
        [BLM_ind_sorted] = sort_local_experts(X,Y,tt,Frol,e_local_ind);
        q = 1;
    end
    
    %% 
    Rbmw = nan(1,T);
    for s = 1:Segm_count
        bM = BLM_list(s,:)'; % 
        % 
        for i = Segments(s,1) : Segments(s,2)
            y_hat = X(i,:)*bM;
            Rbmw(i) = (Y(i)-y_hat)^2;  % 
        end
    end 
    
  %% 
    h = figure('Position',fig_position);
    figname = ['Figure1-AA-BLM-Losses-' figname_suffix '.png'];
    legend_list =[];
    tt    = (H+1:T);
    VcM   = cumsum(Rvvk(tt));    % Кумул. суммы потерь
    
    RbmwcM = cumsum(Rbmw);
    
    legend_list=[];
    
    plot(tt,VcM,':r','LineWidth',2);    hold on
    legend_list=[legend_list {['  AA  ']}];
    
    plot(tt,nVcL,':m','LineWidth',1);    hold on
    legend_list=[legend_list {[' nAA  ']}];
    
    plot(tt,RbmwcM(tt),'b','LineWidth',2);   hold on
    legend_list=[legend_list {['  CE  ']}];
    
    legend(legend_list,'location','west');
    
    title(['Accumulated Losses Means of AA, nAA and CE. ('... 
                                          rcd(figname_suffix) ')']);
    grid on
    saveas(h,figname);
    delete(h);
    
       
    
    Fgen = GMW';
    rr_sigma = sigma;
    
    %% Вспомогательные иллюстрации по локальным экспертам
    %% Считаем и рисуем накопленные потери локальных моделей
    %% Надо не  RRcur!! (это снимок по времени), а заново вычислять все
    %% потери локальных экспертов на полной выборке!!
    draw_local_model_losses = [];
    for dd = draw_local_model_losses   
        RRcur = zeros(T,T);
        % 
        s=1;
        e_local_ind = (Segments(s,1)+H-1:Segments(s,2)); % LocMod in Frol         
        % 
        tt = (Segments(s,1)+H:Segments(s,2));
        [BLM,BLM_ind] = get_best_local_expert(X,Y,tt,Frol,e_local_ind);
        

    end

    
%     
%     save(['reg_models' rcd(figname_suffix) '.mat'],'Frol','Fwrm','Fgen',...
%           'mIndex','H','rr_sigma','figname_suffix');
      
%     save(['WWmatrix' rcd(figname_suffix) '.mat'],'nWWw','nWWv','WWw',...
%                     'WWv','tt','VcM','nVcL','RbmwcM','Segments');  %% ++ Warmuth !
    Q=1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    
    
    
  
    
    q = 1;
    
%% END OF PROGRAM PART

%% Functions:

function y_h = getGammaW(w_cur,F_cur,x_cur)
% 
% 
y_h =(x_cur*F_cur)*w_cur;  
  
  
function y_h = getGammaV(w_cur,F_cur,x_cur,b,eta)
% 
% 
nomi = (x_cur*F_cur)-b;
nomi = -eta*(nomi .^2);
nomi = exp(nomi);
nom  = nomi*w_cur;
deno = (x_cur*F_cur)+b;
deno = -eta*(deno .^2);
deno = exp(deno);
den  = deno*w_cur;
y_h  = 0.25*log(nom/den)/(eta*b);

% 
function mW = get_reg_model(y,X,delta,reg_mode)
    if reg_mode==1
        mW = ridge(y,X,delta);
    else
        mW = regress(y,X);
    end
    
% 
function [r] = rcd(InStrin)
    r = strrep(InStrin,'_','-');    
    

function [LM_ind_sorted] = sort_local_experts(X,Y,tt,Frol,e_local_ind)
     eLossesList = nan(1,length(e_local_ind));  % Потери экспертов на tt
     r2list = nan(length(tt),1);
     
     Xtest = X(tt',:);
     Ytest = Y(tt');
     
     for i = 1:length(e_local_ind)             % По экспертам
         f_current = Frol(:,e_local_ind(i));   % Очередной эксперт
         for j=1:length(tt)
            r2list(j) = get_loss2(Xtest(j,:),f_current,Ytest(j));
         end
         eLossesList(i) = sum(r2list);
     end
     
     [eLossesListSorted,ind] = sort(eLossesList);
     LM_ind_sorted = e_local_ind(ind);
     
        
    
 function [BLM,BLM_ind] = get_best_local_expert(X,Y,tt,Frol,e_local_ind)
    
     eLossesList = nan(1,length(e_local_ind));  % Потери экспертов на tt
     r2list = nan(length(tt),1);
     
     Xtest = X(tt',:);
     Ytest = Y(tt');
     
     for i = 1:length(e_local_ind)             % По экспертам
         f_current = Frol(:,e_local_ind(i));   % Очередной эксперт
         for j=1:length(tt)
            r2list(j) = get_loss2(Xtest(j,:),f_current,Ytest(j));
         end
         eLossesList(i) = sum(r2list);
     end
     
     [eLossesListSorted,ind] = sort(eLossesList);
     %%    BLM,BLM_ind
     BLM_ind =  ind(1);                   
     ind2 = e_local_ind(BLM_ind);
     BLM = Frol(:,e_local_ind(BLM_ind));  
     BLM_1 = Frol(:,BLM_ind);   
              

function [X,Y,Segments,bw,iShift,mIndex,b_param] = ...
                       gen_samples(T,M,K,GMW,sCount,noise_scale,b_upper)
                                                       
        %% 
        X  = randn(T,K);               % N(0,1) 
        Y  = randn(T,1)*noise_scale;   % Noise   
        
        mIndex = ones(T,1); % Active
        bw = fix(T/sCount); % Width
        iShift = 2;      
        
        for s = 1:sCount
            mm = mod(s,M)+1;
            mIndex(1+(s-1)*bw : s*bw)= mm;
        end
        
        change_points = [];
        for t = 1:T-1
            if ~(mIndex(t)==mIndex(t+1))
                change_points = [change_points t];
            end
        end
        %segm_count = length(change_points)+1;
        Segm_count = length(change_points);
        Segments = zeros(Segm_count,3);
        Segments(1,1) = 1;
        for s = 1:Segm_count-1
            Segments(s,2) = change_points(s);
            Segments(s+1,1) = change_points(s)+1;
        end;    
        Segments(Segm_count,2) = T;
        for s = 1:Segm_count
            Segments(s,3) = mIndex(Segments(s,1));
        end
        q=1;
        for t = 1 : T
            m = mIndex(t);  % 
            w = GMW(m,:)';  % 
            % 
            Y(t) = Y(t)+X(t,:)*w;
        end    
        % 
        Ymin = min(Y);
        Ymax = max(Y);
        disp([Ymin Ymax]);  
        b_param = max(abs(Ymin),abs(Ymax));
        if b_param>b_upper
            error('Illegal value of semyinterval!')
        end
        q=1
        
function [L2] = get_loss2(x,f,y) % 
    L2= ((x*f)-y)^2;
  
function [R2Sum,R2list] = evaluate_expert_losses(Ef,Xtest,Ytest)
%%
R2Sum = 0;
R2list = zeros(size(Xtest,1),1); 
for i=1: length(Ytest)
    R2list(i) = get_loss2(Xtest(i,:),Ef,Ytest(i));
end
R2Sum = sum(R2list);

function [R] = explore_experts_on_segment(Segments,SegmInd,GMW,Y,X,Frol,Fwrm,...
                               Rwrm,Rvvk,figname_suffix,H)
%%                                                         
% tt    = (H+1:T);

fig_position = [200 50 700 480]; % Размер окна для графиков                 
%% 1.Извлекаем генерирующую модель и сортируем
K = size(Frol,1);                       % Размерность входных переменных
cur_segment = Segments(SegmInd,:);      % Информация о рабочем сегменте
G           = GMW(cur_segment(3),:);    % Генерирующая модель 
[Gs,ind]    = sort(G);                  % Компоненты генератора (ascended)

%  tt 
tt = (cur_segment(1)+H-1:cur_segment(2)); 


h1 = figure('Position',fig_position);
plot(Frol(ind,tt),'LineWidth',0.5); %
hold on;
plot(Gs,'LineWidth',2); 
E_Frol = mean(Frol(ind,tt)');    
plot(E_Frol,':Y','LineWidth',2);
grid on
ylim([-3,3]);
xlim([1 K]);
title_str = ['GenModel-' num2str(cur_segment(3)) ' + Local experts for S-'...
             num2str(SegmInd) '.'];
title(title_str);
saveas(h1,['Gen+Frol' rcd(figname_suffix) '-S' num2str(SegmInd) '.png']);

%% 
h2 = figure('Position',fig_position);
plot(Fwrm(ind,tt),'LineWidth',0.25); hold on;
% 
plot(Fwrm(ind,tt(1)),':c','LineWidth',2); hold on;   % 
plot(Fwrm(ind,tt(end)),':m','LineWidth',2); hold on; % 
plot(Gs,'-r','LineWidth',2);
title_str = ['GenModel-' num2str(cur_segment(3))...
    ' + WA Aggregators for S-' num2str(SegmInd) '.'];
grid on  
%ylim([-5 5]);
title(title_str);
saveas(h2,['Gen+Wrm' rcd(figname_suffix) '-S' num2str(SegmInd) '.png']); 
R = 0;

R2_list = zeros(1,length(tt));  % Суммарные потери всех Ef экспертов
Wrm_list = zeros(1,length(tt)); % Суммарные потери всех Wrm агрегаторов
EF_R2sum = zeros(length(tt),1);
EWrm_sum = zeros(length(tt),1);
for i = 1:length(tt)        % 
    Ef = Frol(:,tt(i));     % 
    Ewrm = Fwrm(:,tt(i));   % 
    Xtest = X(tt,:);        % 
    Ytest = Y(tt);          % 
  
    [EF_R2sum(i),EF_R2_list] = evaluate_expert_losses(Ef,Xtest,Ytest);
    [EWrm_sum(i),EWrm_list]  = evaluate_expert_losses(Ewrm,Xtest,Ytest);
end

[R2_sorted,ind_s] = sort(EF_R2sum);      
[Wrm_sorted,ind_wrm] = sort(EWrm_sum);   
BestLocal = ind_s(1);
BestWrmth = ind_wrm(1);  
LatestWrmth = length(tt);
EfBest  = Frol(:,BestLocal);     
EfAveraged = mean(Frol')';
WrmBest = Fwrm(:,BestWrmth);     
WrmLatest = Fwrm(:,LatestWrmth); 


[EF_R2,EF_R2_list] =  evaluate_expert_losses(EfBest,Xtest,Ytest);
[EWrmBest,EWrmBest_list] =    evaluate_expert_losses(WrmBest,Xtest,Ytest);
[EWrmLatest,EWrmLatest_list] =    evaluate_expert_losses(WrmLatest,Xtest,Ytest);
[EfAveragedR2,EfAveragedR2_list] = evaluate_expert_losses(EfAveraged,Xtest,Ytest);
[GenR2,GenR2_list]=   evaluate_expert_losses(G',Xtest,Ytest);


h = figure('Position',fig_position);
legend_str = [];
plot(cummean(EF_R2_list),'g','LineWidth',1); hold on;
legend_str = [legend_str; ['Best Local']];
plot(cummean(GenR2_list),'-b','LineWidth',1); hold on;
legend_str = [legend_str; ['Generator ']];
plot(cummean(EWrm_list),'-r','LineWidth',1); hold on;
legend_str =[legend_str; ['Best Aggr.']];
plot(cummean(EWrmLatest_list),':c','LineWidth',1); hold on;
legend_str =[legend_str; ['LatestAggr']];
plot(cummean(EfAveragedR2_list),':m','LineWidth',1); hold on;
legend_str =[legend_str; ['EfAveraged']];
% EWrmLatest,EWrmLatest_list
%plot(cummean(EWrmLatest_list),':c','LineWidth',1); hold on;
%legend_str =[legend_str; ['EWrmLatest']];
title(['Loss cummeans for Gen and best(Ef,WRM)-S' num2str(SegmInd)]);

legend(legend_str,'Location','Best');
grid on;
xlim([0 length(tt)]);
%saveas(h,['LossCummeans-S' num2str(SegmInd) '-' rcd(figname_suffix) '.fig']);
saveas(h,['LossCummeans-S' num2str(SegmInd) '-' rcd(figname_suffix) '.png']);
 q=1;   

 