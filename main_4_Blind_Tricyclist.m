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
comparison_EKF = 1;         % 1: compare with EKF / 0: no comparison
comparison_UKF = 1;         % 1: compare with UKF / 0: no comparison
comparison_SIR = 1;         % 1: compare with SIR / 0: no comparison

numParticles = 7000;
numEnsembles = numParticles;
numPartiAKKF = 1000;
numParticlSIR = numParticles;
numMixturetemp = 100;
numMC = 100;

rng(42);

%% Define a scenario
load blindtricyc03_example_new      % clear and load only new data

% Parameters
numStep = size(t,1);                % retrieve number of samples
numMGR = size(Xvec,1);              % retrieve number of merry-go-rounds
nw = size(Q,1);                     % retrieve number of process noise elements

n = 3+2*numMGR;                   % compute number of states
nm = numMGR;                        % size of the mearuements size(R,1)

wbar = zeros(nw,1);                 % assign a priori process noise vector
merging_thres = chi2inv(0.05,n);    % threshold for merging mixtures 0.995

% P0 = diag([7.5^2; 7.5^2; (pi/4)^2; (pi/3)^2; (pi/3)^2;...
%                 (7.427e-03)^2; (7.427e-03)^2]); % To test with EKF and UKF

P0 = diag([18.75^2; 18.75^2; (5*pi/8)^2; (5*pi/6)^2; (5*pi/6)^2;...
                (1.857e-02)^2; (1.857e-02)^2]); % Real test

iflagobsmat = 0;                    % flag to not form observability matrix

% UT parameters
numSigma = 2*n+1;
alpha = 0.01;
beta = 2.0;
kappa = 0;
lambda = alpha^2*(n+kappa)-n;

%% For evaluation
% Error and RMSE
errorall = zeros(8,n,numStep,numMC);
errornorm = zeros(8,numStep,numMC);
rmse = zeros(8,numStep);
armse = zeros(8,1);

% Chi-square test
chisqall = zeros(8,numStep,numMC);
chinorm = zeros(8,numStep);
achi = zeros(8,1);

chithres = chi2inv(0.9999,n);
chicntall = zeros(8,numStep,numMC);
chicntfin = zeros(8,numStep,1);
chifrac = zeros(8,numStep,1);
chifracavg = zeros(8,1);

% Time-related
elpsdtime = zeros(8,numMC);

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
    [trueState,meas,yaltkhist_true,wkhist,initial,R] = ...
                        blindtricyc03_siml(x0,ukhist,t,bw,br,...
                        Xvec,Yvec,rhovec,sigmavec,...
                        imykflaghist,P0,Q,...
                        iflagobsmat,[]);
    % For evaluation
    error = zeros(8,n,numStep);
    poserror = zeros(8,numStep);
    chisq = zeros(8,numStep); % chi-squared statistic storage
    elpst = zeros(8,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);

    %% For plot
    if figure_view
        labels = {'PGM-DS','PGM-DU','PGM1','PGM1-UT','AKKF','EKF','UKF','SIR'};
        markers = {'s','^','d','v','p','>','x','+'};
        colors  = {'r','g','#0072BD','#EDB120','m','#7E2F8E','#4DBEEE','#77AC30'};
        remap_columnwise = [1 5 2 6 3 7 4 8];
        marker_indices = 1:5:length(t);

        fig2 = figure; hold on; grid on;
        h2 = gobjects(1,8);
        h_true = plot(trueState(:,1),trueState(:,2),'--ok',...
                      'DisplayName','True',...
                      'LineWidth',1,...
                      'MarkerSize',6,...
                      'MarkerIndices',marker_indices);
        xlabel('East position (m)');
        ylabel('North position (m)');
        axis('equal')
    
        fig3 = figure; hold on; grid on;
        h3 = gobjects(1,8);
        xlabel('Time (sec)');
        ylabel('|Position estimation error| (m)');
        xlim([0 141]);
        set(gca,'YScale','log')

        % Pre-fill dummy handles for other algorithms
        for k = 1:8
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
        epsilon = 5;            % distance to determine if a point is a core point 50
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial.*ones(n,numParticles)+sqrt(P0)*randn(n,numParticles);
        
        for p = 1:numParticles
            [particle_meas(:,p,1),~,~] = blindtricyc03_haltfunct(particles(:,p,1),0,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
        end 
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
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
                [particles(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(particles(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
                
                particles(:,p,i) = particles(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,...
                numMixture,~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);

            %% Merging
            [particles,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            for p = 1:numParticles
                [particle_meas(:,p,i),~,~] = blindtricyc03_haltfunct(particles(:,p,i),i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
                particle_meas(:,p,i) = particle_meas(:,p,i)+sqrt(R)*randn(nm,1);
            end 
    
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);

            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end   
        end
        
        elpst(fil_idx) = toc(tStart1);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% PGM_DU
    if comparison_PGM_DU
        %% Filter parameters
        fil_idx = 2;
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point
        epsilon = 5;            % distance to determine if a point is a core point 50
        
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
        particles(:,:,1) = initial.*ones(n,numParticles)+sqrt(P0)*randn(n,numParticles);
            
        for p = 1:numParticles
            [particle_meas(:,p,1),~,~] = blindtricyc03_haltfunct(particles(:,p,1),0,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
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
                [particles(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(particles(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
                            
                particles(:,p,i) = particles(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,...
                numMixture,~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);
    
            %% Merging
            [particles,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            funmeas = @(x,v) blindtricyc03_haltfunct(x,i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
    
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);

            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
        end
        
        elpst(fil_idx) = toc(tStart2);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
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
        numMixture_max = 3;     % number of Gaussian mixtures
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
        
        %% First time step
        particles(:,:,1) = initial.*ones(n,numParticles)+sqrt(P0)*randn(n,numParticles);
        
        for p = 1:numParticles
            [particle_meas(:,p,1),~,~] = blindtricyc03_haltfunct(particles(:,p,1),0,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
        end 
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
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
                [particles(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(particles(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
    
                particles(:,p,i) = particles(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [particles,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            for p = 1:numParticles
                [particle_meas(:,p,i),~,~] = blindtricyc03_haltfunct(particles(:,p,i),i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
                particle_meas(:,p,i) = particle_meas(:,p,i)+sqrt(R)*randn(nm,1);
            end 
    
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);

            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end
        end
        
        elpst(fil_idx) = toc(tStart3);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
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
        numMixture_max = 3;     % number of Gaussian mixtures
        
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
        particles(:,:,1) = initial.*ones(n,numParticles)+sqrt(P0)*randn(n,numParticles);
            
        for p = 1:numParticles
            [particle_meas(:,p,1),~,~] = blindtricyc03_haltfunct(particles(:,p,1),0,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,ccovar,cweight,~,~] = ...
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
                [particles(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(particles(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
    
                particles(:,p,i) = particles(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [particles,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            funmeas = @(x,v) blindtricyc03_haltfunct(x,i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
    
            [particles,estState,temp_particle_var,...
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);

            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
        end
        
        elpst(fil_idx) = toc(tStart4);
        
        if figure_view
            fil_idx = 4;

            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
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
        AKKF.N_P = numPartiAKKF;
    
        AKKF.c = 1;
        AKKF.lambda = 10^(-3);      % kernel normisation parameter
        AKKF.lambda_KKF = 10^(-3);  % kernel normisation parameter for Kernel Kalman filter gain calculation
        AKKF.poly_para_b = 10^(3);  % polynomial triack scale for the bearing tracking
        AKKF.Var_Gaussian = R;
    
        %% Initialisation
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
      
        %% First time step
        AKKF.X_P(:,:,1) = initial.*ones(n,numPartiAKKF)+sqrt(P0)*randn(n,numPartiAKKF);
    
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
        
        for p = 1:numPartiAKKF
            [AKKF.Z_P(:,p,1),~,~] = blindtricyc03_haltfunct(AKKF.X_P(:,p,1),0,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
        end 
        
        estState(:,1) = mean(AKKF.X_P(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        %% Main loop
        tStart5 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numPartiAKKF
                [AKKF.X_P(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(AKKF.X_P(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
    
                AKKF.X_P(:,p,i) = AKKF.X_P(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
    
            [AKKF] = AKKF_predict(AKKF,i);
    
            %% Update
            for p = 1:numPartiAKKF
                [AKKF.Z_P(:,p,i),~,~] = blindtricyc03_haltfunct(AKKF.X_P(:,p,i),i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
                AKKF.Z_P(:,p,i) = AKKF.Z_P(:,p,i)+sqrt(R)*randn(nm,1);
            end 
    
            [AKKF] = AKKF_update(meas',AKKF,R,i);
            [AKKF] = AKKF_proposal(AKKF,i);
    
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
            
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
        end
        
        elpst(fil_idx) = toc(tStart5);
        
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end
    
    %% EKF
    if comparison_EKF
        %% Filter parameters
        fil_idx = 6;

        %% Initialisation
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        x = initial;
        P = P0;
        
        estState(:,1) = x;
        temp_particle_var(:,:,1) = P;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        %% Main loop
        tStart6 = tic; 
        
        for i = 2:numStep
            %% Prediction
            [x,F,G] = blindtricyc03_ffunct(x,wbar,i-2,0,ukhist,t,bw);
            P = F*P*F'+G*Q*G';
        
            %% Update
            [y,H] = blindtricyc03_haltfunct(x,i-1,0,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
            K = P*H'/(R+H*P*H');
            x = x+K*(meas(i,:)'-y);
            P = (eye(n)-K*H)*P*(eye(n)-K*H)'+K*R*K';
            estState(:,i) = x;
        
            %% For evaluation
            error(fil_idx,:,i) = x-trueState(i,:)';
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(P\error(fil_idx,:,i)');
        end
    
        elpst(fil_idx) = toc(tStart6);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(fil_idx,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end
    
    %% UKF
    if comparison_UKF
        %% Filter parameters
        fil_idx = 7;

        %% Initialisation
        xs = zeros(n,numSigma,numStep);
        zs = zeros(nm,numSigma,numStep);
        wm = zeros(numSigma,1);
        wc = zeros(numSigma,1);
        xn = zeros(n,1);
        Pn = zeros(n);
        zn = zeros(nm,1);
        Pzz = zeros(nm);
        Pxz = zeros(n,nm);
        
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        x = initial;
        P = P0;
    
        wm(1) = lambda/(n+lambda);
        wc(1) = lambda/(n+lambda)+(1-alpha^2+beta);
        
        for j = 2:numSigma   
            wm(j) = 1/(2*(n+lambda));
            wc(j) = wm(j);
        end
    
        estState(:,1) = x;
        temp_particle_var(:,:,1) = P;
        
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        %% Main loop
        tStart7 = tic; 
    
        for i = 2:numStep
            %% Sigma points
            sqrt_P = chol((n+lambda)*P,'lower'); % Square root of scaled covariance
            xs(:,1,i) = x;
            for j = 1:n
                xs(:,j+1,i) = x+sqrt_P(:,j);
                xs(:,j+n+1,i) = x-sqrt_P(:,j);
            end
    
            %% Prediction
            for j = 1:numSigma
                [xs(:,j,i),~,G] = blindtricyc03_ffunct(xs(:,j,i),...
                                    wbar,i-2,0,ukhist,t,bw);
                xn = xn+wm(j)*xs(:,j,i);
            end
    
            for j = 1:numSigma
                Pn = Pn+wc(j)*(xs(:,j,i)-xn)*(xs(:,j,i)-xn)';
            end
            Pn = Pn+G*Q*G';   
            
            %% Update
            for j = 1:numSigma
                [zs(:,j,i),~] = blindtricyc03_haltfunct(xs(:,j,i),i-1,1,br,...
                            Xvec,Yvec,rhovec,imykflaghist,meas);
                zn = zn+wm(j)*zs(:,j,i);
            end
            
            for j = 1:numSigma
                Pzz = Pzz+wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
            end
            Pzz = Pzz+R;
        
            for j = 1:numSigma
                Pxz = Pxz+wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
            end
        
            K = Pxz/Pzz;
            x = xn+K*(meas(i,:)'-zn);
            P = Pn-K*Pzz*K';
    
            estState(:,i) = x;
        
            %% For evaluation
            error(fil_idx,:,i) = abs(x-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(P\error(fil_idx,:,i)'); 
    
            %% Initialisation
            xn = zeros(n,1);
            Pn = zeros(n);
            zn = zeros(nm,1);
            Pzz = zeros(nm);
            Pxz = zeros(n,nm);
        end
        
        elpst(fil_idx) = toc(tStart7);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(7,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% SIR
    if comparison_SIR
        %% Filter parameters
        fil_idx = 8;
        numEffparticles = 5000;
    
        %% Initialisation
        particles = zeros(n,numParticlSIR,numStep);
        particle_meas = zeros(nm,numParticlSIR,numStep);
        particle_w = zeros(numParticlSIR,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial.*ones(n,numParticlSIR)+sqrt(P0)*randn(n,numParticlSIR);        
        particle_w(:,1) = 1/numParticlSIR*ones(numParticlSIR,1);
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(fil_idx,1) = sqrt(error(fil_idx,1,1)^2+error(fil_idx,2,1)^2);
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        %% Main loop
        tStart8 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticlSIR
                [particles(:,p,i),~,G] = ...
                    blindtricyc03_ffunct(particles(:,p,i-1),wbar,i-2,0,...
                    ukhist,t,bw);
    
                particles(:,p,i) = particles(:,p,i)+G*sqrtm(Q)*randn(nw,1);
            end
                    
            for p = 1:numParticlSIR
                [particle_meas(:,p,i),~,~] = blindtricyc03_haltfunct(particles(:,p,i),i-1,1,br,...
                        Xvec,Yvec,rhovec,imykflaghist,meas);
                residual = meas(i,:)'-particle_meas(:,p,i);
                particle_w(p,i) = 1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
            end

            %% Resample
            if all(particle_w(:,i) == 0)
                particle_w(:,i) = 1/numParticlSIR;
            end

            particle_w(:,i) = particle_w(:,i)/sum(particle_w(:,i));
            particles(:,:,i) = resampleEff(particle_w(:,i),...
                                    particles(:,:,i),numEffparticles);

            estState(:,i) = mean(particles(:,:,i),2);
    
            %% For evaluation
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
    
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(fil_idx,i) = sqrt(error(fil_idx,1,i)^2+error(fil_idx,2,i)^2);
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
        end
        
        elpst(fil_idx) = toc(tStart8);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(estState(1,:),estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig3);
            h3(fil_idx) = plot(t,poserror(8,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end
   
    %% End of parfor
    errorall(:,:,:,mc) = error;
    poserrorall(:,:,mc) = poserror;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    cmeanall(:,:,:,:,mc) = cmeanmc;

    if figure_view
        labels = {'PGM-DS','AKKF','PGM-DU','EKF','PGM1','UKF','PGM1-UT','SIR'};

        figure(fig2);
        legend(h2(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([0 55])

        ax_main = gca;
        ax_true = axes('Position',get(ax_main,'Position'),...
                       'Color','none',...
                       'XColor','none','YColor','none');
        lgd_true = legend(ax_true,h_true,'True','Location','southeast');
        set(lgd_true,'Box','off');
    
        figure(fig3);
        legend(h3(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([0 255])
    end
end

%% Evaluations
% For plot
labels = {'PGM-DS','AKKF','PGM-DU','EKF','PGM1','UKF','PGM1-UT','SIR'};
markers = {'s','^','d','v','p','>','x','+'};
colors  = {'r','g','#0072BD','#EDB120','m','#7E2F8E','#4DBEEE','#77AC30'};
remap_columnwise = [1 5 2 6 3 7 4 8];
tlim = 141;
marker_indices = 1:5:length(t);

comparison_flags = [ ...
    comparison_PGM_DS,...
    comparison_PGM_DU,...
    comparison_PGM1,...
    comparison_PGM1_UT,...
    comparison_AKKF,...
    comparison_EKF,...
    comparison_UKF,...
    comparison_SIR];

comparison_flags =  1 - comparison_flags;

% Root means squared error (RMSE)
for fil = 1:8
    for i = 1:numStep
        for mc = 1:numMC
            errornorm(fil,i,mc) = sqrt(errorall(fil,1,i,mc)^2+errorall(fil,2,i,mc)^2); % Position error
        end
    end
end

for fil = 1:8
    for i = 1:numStep
        rmse(fil,i) = sqrt(sum(errornorm(fil,i,1:numMC).^2)/numMC);
    end
end

% Averaged RMSE
for fil = 1:8
    armse(fil) = sum(rmse(fil,:))/numStep;
end

% Plot RMSE
figure; hold on; grid on;
h = gobjects(1,8);  % Preallocate

for k = 1:8
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

xlim([0 tlim])
ylim([0 250])
xlabel('Time (sec)')
ylabel('Position RMSE (m)')

set(gca, 'YScale', 'log');

hold off;

% Average terminate position error
for fil = 1:8
    lastpos = poserrorall(fil,end,:); % Position error
    lastpos = reshape(lastpos,[1,numMC]);

    % atermerr(fil) = mean(lastpos);
    atermerr(fil) = sqrt(sum(lastpos.^2)/numMC);
end

atermerr = atermerr';

% Chi-squared
for fil = 1:8
    for i = 1:numStep
        chinorm(fil,i) = sum(chisqall(fil,i,:))/(n*numMC);
    end
end

for fil = 1:8
    achi(fil) = sum(chinorm(fil,:))/numStep;
end

% Counting Chi-squared
for fil = 1:8
    for i = 1:numStep
        for mc = 1:numMC
            if chisqall(fil,i,mc) > chithres
                chicntall(fil,i,mc) = 1;
            end
        end
    end
end

for fil = 1:8
    for i = 1:numStep
        chicntfin(fil,i) = sum(chicntall(fil,i,:) == 1);
        chifrac(fil,i) = chicntfin(fil,i)/numMC;
    end
end

for fil = 1:8
    chifracavg(fil) = mean(chifrac(fil,:),2);
end

% Plot Chisq percentage cases
figure; hold on; grid on;
h = gobjects(1,8);  % Preallocate

for k = 1:8
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

xlim([0 tlim])
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
save('4_Blind_Tricyclist_MC')
