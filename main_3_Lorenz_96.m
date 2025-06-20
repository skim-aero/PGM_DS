%% DBSCAN-based Particle Gaussian Mixture Filters
% Sukkeun Kim (s.kim.aero@gmail.com)

clc; clear; close all;
addpath(genpath('functions'));

warning('off','all')
warning

%% Simulation settings
figure_view = 1;            % 1: only when run with for (not parfor)
comparison_PGM_DS = 1;      % 1: compare with PGM_DS  / 0: no comparison
comparison_PGM_DU = 1;      % 1: compare with PGM_DU  / 0: no comparison
comparison_PGM1 = 1;        % 1: compare with PGM1  / 0: no comparison
comparison_PGM1_UT = 1;     % 1: compare with PGM1_UT  / 0: no comparison
comparison_AKKF = 1;        % 1: compare with AKKF / 0: no comparison
comparison_EnKF = 1;        % 1: compare with EnKF / 0: no comparison
comparison_SIR = 1;         % 1: compare with SIR / 0: no comparison

numParticles = 2000;
numEnsembles = numParticles;
numMixturetemp = 10;
numMC = 100;

rng(42);

%% Define a scenario
% Time
dt = 0.05;
tf = 10-dt;
t = 0:dt:tf;
numStep = length(t);

% Parameters
n = 40;         % dimension of the state
nm = 20;        % dimension of the measurement 

Param_F = 8;
Q = 10E-3;
R = 10E-3;
merging_thres = chi2inv(0.995,n); % threshold for merging mixtures

trueinitial = Param_F*ones(n,1);
P0 = 10E-4*eye(n);

initial = trueinitial;

% UT parameters
numSigma = 2*n+1;
alpha = 1e-3;
beta = 2;
kappa = 0;
lambda = alpha^2*(n+kappa)-n;

% System/measurement equations
funsys = @(x,w,k,F) Lorenz4DP(x,w,k,F);

H = zeros(nm,n);
for i = 1:nm
    for j = 1:n
        temp = 2*i-1; 
        if j == temp
            H(i,j) = 1;
        else
            H(i,j) = 0;
        end
    end
end

funmeas = @(x,v) H*x+v;
jacobianF = @(x) Lorenz4DJ(x);

G = eye(n);

%% For evaluation
% Error and RMSE
errorall = zeros(7,n,numStep,numMC);
errornorm = zeros(7,numStep,numMC);
rmse = zeros(7,numStep);
armse = zeros(7,1);

% Chi-square test
chisqall = zeros(7,numStep,numMC);
chinorm = zeros(7,numStep);
achi = zeros(7,1);

chithres = chi2inv(0.9999,n);
chicntall = zeros(7,numStep,numMC);
chicntfin = zeros(7,numStep,1);
chifrac = zeros(7,numStep,1);
chifracavg = zeros(7,1);

% Time-related
elpsdtime = zeros(7,numMC);

% Cluster log
cmeanall = zeros(4,n,numMixturetemp,numStep,numMC);

%% Monte Carlo simulation
parpool(4) % Limit the workers to half of the CPU due to memory...
parfevalOnAll(@warning,0,'off','all'); % Temporary error off
parfor mc = 1:numMC
% for mc = 1:numMC 
    disptemp = ['Monete Carlo iteration: ',num2str(mc)];
    disp(disptemp)

    % Generate true states
    initial_wnoise = trueinitial+sqrt(P0)*randn(n,1);
    trueState = Lorenz4D(tf,Q,initial_wnoise,Param_F);

    % Generate measurements
    measNoise = sqrt(R)*randn(nm,numStep);
    meas = funmeas(trueState,measNoise);

    % For evaluation
    error = zeros(7,n,numStep);
    chisq = zeros(7,numStep); % chi-squared statistic storage
    elpst = zeros(7,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);

    %% For plot
    if figure_view
        labels = {'PGM-DS','PGM-DU','PGM1','PGM1-UT','AKKF','EnKF','SIR'};
        markers = {'s','^','d','v','p','h','+'};
        colors  = {'r','g','#0072BD','#EDB120','m','c','#77AC30'};
        remap_columnwise = [1 5 2 6 3 7 4];
        marker_indices = 1:4:length(t);

        fig2 = figure; hold on; grid on;
        h2 = gobjects(1,7);
        h_true = plot(t,trueState(1,:),'--ok',...
                'DisplayName','True',...
                'LineWidth',1,...
                'MarkerSize',6,...
                'MarkerIndices',marker_indices);
        xlim([0 tf]);
        xlabel('Time (sec)');
        ylabel('Estimated state (first)');
    
        fig3 = figure; hold on; grid on;
        h3 = gobjects(1,7);
        xlim([0 tf]);
        xlabel('Time (sec)');
        ylabel('|Estimation error (first)|');

        % Pre-fill dummy handles for other algorithms
        for k = 1:7
            h2(k) = plot(NaN,NaN,'Marker',markers{k},...
                'Color',colors{k},'LineStyle','-',...
                'MarkerSize',6,'LineWidth',1,...
                'DisplayName',labels{k},'Visible','off');

            h3(k) = plot(NaN,NaN,'Marker',markers{k},...
                'Color',colors{k},'LineStyle','-',...
                'MarkerSize',6,'LineWidth',1,...
                'DisplayName',labels{k},'Visible','off');
        end
    end
   
    %% PGM-DS
    if comparison_PGM_DS
        %% Filter parameters
        fil_idx = 1;
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point
        epsilon = 100;          % distance to determine if a point is a core point 100
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
        
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
       
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
     
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,~,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
        
        %% Main loop
        tStart1 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = Lorenz4DP(particles(:,p,i-1),Q,[],Param_F);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);

            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
            
            %% Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end 
    
            if rem(i,20) == 0
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           1,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,0,0,0,0,0,0);
            else
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           0,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,0,0,0,0,0,0);
            end

            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end   
        end
        
        elpst(fil_idx) = toc(tStart1);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% PGM-DU
    if comparison_PGM_DU
        %% Filter parameters
        fil_idx = 2;
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point
        epsilon = 100;          % distance to determine if a point is a core point 100
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        wm = zeros(numSigma,1);
        wc = zeros(numSigma,1);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
        
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,~,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
    
        %% Main loop
        tStart2 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = Lorenz4DP(particles(:,p,i-1),Q,[],Param_F);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            if rem(i,20) == 0
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           1,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,...
                                           1,numSigma,lambda,wc,wm,funmeas);
            else
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           0,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,...
                                           1,numSigma,lambda,wc,wm,funmeas);
            end
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
        end
    
        elpst(fil_idx) = toc(tStart2);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% PGM1
    if comparison_PGM1
        %% Filter parameters
        fil_idx = 3;
        numMixture_max = 2;     % number of Gaussian mixtures
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
        
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
       
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
     
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,~,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
    
        %% Main loop
        tStart3 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = Lorenz4DP(particles(:,p,i-1),Q,[],Param_F);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end 
    
            if rem(i,20) == 0
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           1,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,0,0,0,0,0,0);
            else
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           0,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,0,0,0,0,0,0);
            end
    
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end   
        end
        
        elpst(fil_idx) = toc(tStart3);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% PGM1_UT
    if comparison_PGM1_UT
        %% Filter parameters
        fil_idx = 4;
        numMixture_max = 2;     % number of Gaussian mixtures
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        wm = zeros(numSigma,1);
        wc = zeros(numSigma,1);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
        
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,~,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
    
        %% Main loop
        tStart4 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = Lorenz4DP(particles(:,p,i-1),Q,[],Param_F);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            if rem(i,20) == 0
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           1,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,...
                                           1,numSigma,lambda,wc,wm,funmeas);
            else
                [particles,estState,temp_particle_var,...
                 cmean,~,cweight] = ...
                               PGM_DS_update(i,n,nm,numStep,numParticles,...
                                           numMixture,numMixture_max,...
                                           0,cweight,idx,meas,likelih,R,...
                                           particles,particle_clust,...
                                           particle_meas,particle_mean,...
                                           particle_var,estState,...
                                           temp_particle_var,...
                                           1,numSigma,lambda,wc,wm,funmeas);
            end
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
        end
    
        elpst(fil_idx) = toc(tStart4);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% AKKF
    if comparison_AKKF
        %% Filter parameters
        fil_idx = 5;
        AKKF.kernel = 1;            % "Quadratic" "Quartic" "Gaussian"
        AKKF.N_P = numParticles;
    
        AKKF.c = 1; %
        AKKF.lambda = 10^(-3);      % kernel normisation parameter
        AKKF.lambda_KKF = 10^(-3);  % kernel normisation parameter for Kernel Kalman filter gain calculation
        AKKF.poly_para_b = 10^(3);  % polynomial triack scale for the bearing tracking
        AKKF.Var_Gaussian = R;
    
        %% Initialisation
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        AKKF.X_P(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        % Data space
    %     AKKF.X_P(:,:,1) = mgd(AKKF.N_P,4,Tar.X(:,1),Sys.P0).'; % state particles
        
        % Kernel space
        AKKF.W_minus(:,1) = ones(AKKF.N_P,1)/AKKF.N_P;
        AKKF.S_minus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
        
        AKKF.W_plus(:,1) = ones(AKKF.N_P,1)/AKKF.N_P;
        AKKF.S_plus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
        
        AKKF.X_P_proposal(:,:,1) =  AKKF.X_P(:,:,1);
        
        AKKF.X_est(:,1) = AKKF.X_P(:,:,1) * AKKF.W_plus(:,1); %state mean
        AKKF.X_C(:,:,1) =   AKKF.X_P(:,:,1) * AKKF.S_plus(:,:,1) * AKKF.X_P(:,:,1).'; %state covariance
        
        for p = 1:numParticles
            AKKF.Z_P(:,p,1) = funmeas(AKKF.X_P(:,p,1),0);
        end 
    
        estState(:,1) = mean(AKKF.X_P(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        %% Main loop
        tStart5 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                AKKF.X_P(:,p,i) = Lorenz4DP(AKKF.X_P_proposal(:,p,i-1),Q,[],Param_F);
            end
    
            [AKKF] = AKKF_predict(AKKF,i);
    
            %% Update
            for p = 1:numParticles
                AKKF.Z_P(:,p,i) = funmeas(AKKF.X_P(:,p,i),sqrt(R)*randn(nm,1));
            end 
    
            if rem(i,20) == 0
                [AKKF] = AKKF_update(meas,AKKF,R,i);
                [AKKF] = AKKF_proposal(AKKF,i);
            else
                AKKF.W_minus(:,i) = AKKF.W_minus(:,i-1);
                AKKF.S_minus(:,:,i) = AKKF.S_minus(:,:,i-1);
                
                AKKF.W_plus(:,i) = AKKF.W_plus(:,i-1);
                AKKF.S_plus(:,:,i) = AKKF.S_plus(:,:,i-1);
                
                AKKF.X_P_proposal(:,:,i) = AKKF.X_P(:,:,i);
                
                AKKF.X_est(:,i) = AKKF.X_P(:,:,i) * AKKF.W_plus(:,i); %state mean
                AKKF.X_C(:,:,i) =   AKKF.X_P(:,:,i) * AKKF.S_plus(:,:,i) * AKKF.X_P(:,:,i).'; %state covariance
            end
    
            estState(:,i) = mean(AKKF.X_P(:,:,i),2);
    
            %% For evaluation
            for j = 1:n
                for k = 1:n
                    if j == k
                        temp_particle_var(j,k,i) = var(AKKF.X_P(j,:,i));
                    else
                        temp = cov(AKKF.X_P(j,:,i),AKKF.X_P(k,:,i));
                        temp_particle_var(j,k,i) = temp(1,2);
                    end
                end
            end
            
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
        end
        
        elpst(fil_idx) = toc(tStart5);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            fil_idx = 5;

            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% EnKF
    if comparison_EnKF
        %% Filter parameters
        fil_idx = 6;

        %% Initialisation
        ensembles = zeros(n,numEnsembles,numStep);
        ensemble_mean = zeros(n,numStep);
        ensemble_var = zeros(n,n,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step  
        ensembles(:,:,1) = initial+sqrt(P0)*randn(n,numEnsembles);
        ensemble_var(:,:,1) = P0;
        
        estState(:,1) = mean(ensembles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        %% Main loop
        tStart6 = tic;

        [estState,ensemble_var] = EnKF(numStep,n,nm,meas,numEnsembles,...
                                       estState,ensembles,ensemble_var,ensemble_mean,...
                                       Q,R,[],H,[],funsys,funmeas,20);
        
        %% For evaluation
        for i = 2:numStep
            temp_particle_var(:,:,i) = ensemble_var(:,:,i);
    
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
        end
    
        elpst(fil_idx) = toc(tStart6);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% SIR
    if comparison_SIR
        %% Filter parameters
        fil_idx = 7;
        numEffparticles = 400;
    
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        weights = zeros(numParticles,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial*ones(1,numParticles)+sqrt(P0)*randn(n,numParticles);
        weights(:,1) = 1/numParticles*ones(numParticles,1);
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        %% Main loop
        tStart7 = tic; 

        [estState,particles,weights] = ...
                SIR(numStep,n,nm,meas,numParticles,estState,particles,weights,...
                    Q,R,[],[],funsys,funmeas,numEffparticles,20);
    
        %% For evaluation
        for i = 2:numStep
            for j = 1:n
                for k = 1:n
                    if j == k
                        temp_particle_var(j,k,i) = var(particles(j,:,i));
                    else
                        temp = cov(particles(j,:,i),particles(k,:,i));
                        temp_particle_var(j,k,i) = temp(1,2);
                    end
                end
            end
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
        end
        
        elpst(fil_idx) = toc(tStart7);
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end
   
    %% End of parfor
    errorall(:,:,:,mc) = error;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    cmeanall(:,:,:,:,mc) = cmeanmc;

    if figure_view
        labels = {'PGM-DS','AKKF','PGM-DU','EnKF','PGM1','SIR','PGM1-UT'};

        figure(fig2);
        legend(h2(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([-10 18])
        
        ax_main = gca;
        ax_true = axes('Position',get(ax_main,'Position'),...
                       'Color','none',...
                       'XColor','none','YColor','none');
        lgd_true = legend(ax_true,h_true,'True','Location','southeast');
        set(lgd_true,'Box','off');
    
        figure(fig3);
        legend(h3(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([0 12])
    end
end

%% Evaluations
% For plot
labels = {'PGM-DS','AKKF','PGM-DU','EnKF','PGM1','SIR','PGM1-UT'};
markers = {'s','^','d','v','p','h','+'};
colors  = {'r','g','#0072BD','#EDB120','m','c','#77AC30'};
remap_columnwise = [1 5 2 6 3 7 4];
marker_indices = 1:1:length(t);

comparison_flags = [ ...
    comparison_PGM_DS,...
    comparison_PGM_DU,...
    comparison_PGM1,...
    comparison_PGM1_UT,...
    comparison_AKKF,...
    comparison_EnKF,...
    comparison_SIR];

comparison_flags =  1 - comparison_flags;

% Root means squared error (RMSE)
for fil = 1:7
    for i = 1:numStep
        for mc = 1:numMC
            errornorm(fil,i,mc) = vecnorm(errorall(fil,:,i,mc));
        end
    end
end

for fil = 1:7
    for i = 1:numStep
        rmse(fil,i) = sqrt(sum(errornorm(fil,i,1:numMC).^2)/numMC);
    end
end

% Averaged RMSE
for fil = 1:7
    armse(fil) = sum(rmse(fil,:))/numStep;  % PGM-UT/PGM/UKF/SIR
end

% Plot RMSE
figure; hold on; grid on;
h = gobjects(1,7);  % Preallocate

for k = 1:7
    if comparison_flags(k)
        h(k) = plot(NaN,NaN,...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'Visible','off');
    else
        h(k) = plot(t,rmse(k,:),...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'MarkerIndices',marker_indices);
    end
end

legend(h(remap_columnwise),labels,'NumColumns',4,'Location','north');

xlim([0 tf])
ylim([0 52])
xlabel('Time (sec)')
ylabel('RMSE')
hold off;

% Chi-Square
for fil = 1:7
    for i = 1:numStep
        chinorm(fil,i) = sum(chisqall(fil,i,:))/(n*numMC);
    end
end

for fil = 1:7
    achi(fil) = sum(chinorm(fil,:))/numStep;
end

% Counting Chi-squared
for fil = 1:7
    for i = 1:numStep
        for mc = 1:numMC
            if chisqall(fil,i,mc) > chithres
                chicntall(fil,i,mc) = 1;
            end
        end
    end
end

for fil = 1:7
    for i = 1:numStep
        chicntfin(fil,i) = sum(chicntall(fil,i,:) == 1);
        chifrac(fil,i) = chicntfin(fil,i)/numMC;
    end
end

for fil = 1:7
    chifracavg(fil) = mean(chifrac(fil,:),2);
end

% Plot Chisq percentage cases
figure; hold on; grid on;
h = gobjects(1,7);  % Preallocate

for k = 1:7
    if comparison_flags(k)
        h(k) = plot(NaN,NaN,...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'Visible','off');
    else
        h(k) = plot(t,chifrac(k,:),...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'MarkerIndices',marker_indices);
    end
end

legend(h(remap_columnwise),labels,'NumColumns',4,'Location','north');

xlim([0 tf])
ylim([0 1.15])
xlabel('Time (sec)')
ylabel('Cases exceeding 99.99% \chi^2 bound (%)');
ax = gca;

yticks = get(ax,'YTick');
yticklabels = arrayfun(@(y) sprintf('%.0f%%',y * 100),yticks,'UniformOutput',false);
set(ax,'YTickLabel',yticklabels);
hold off;

elpadtavg = mean(elpsdtime,2);

%% Save variables
save('3_Lorenz_96_MC')