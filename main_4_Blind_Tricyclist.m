%% Particle Gaussian Mixture Filter with DBSCAN (and with UT)
% Sukkeun Kim (Sukkeun.Kim@cranfield.ac.uk)

clc; clear; close all;
addpath(genpath('functions'));

warning('off','all')
warning

%% Simulation settings
figure_view = 1;
comparison_PGM_DS = 1;      % 1: compare with PGM_DS  / 0: no comparison
comparison_PGM_DU = 1;      % 1: compare with PGM_DU  / 0: no comparison
comparison_PGM1 = 0;        % 1: compare with PGM1  / 0: no comparison
comparison_PGM1_UT = 0;     % 1: compare with PGM1_UT  / 0: no comparison
comparison_AKKF = 0;        % 1: compare with AKKF / 0: no comparison
comparison_EnKF = 1;        % 1: compare with EnKF / 0: no comparison
comparison_EKF = 1;         % 1: compare with EKF / 0: no comparison
comparison_UKF = 1;         % 1: compare with UKF / 0: no comparison
comparison_SIR = 1;         % 1: compare with SIR / 0: no comparison

numParticles = 7000;
numEnsembles = numParticles;
numPartiAKKF = 1000;
numParticlSIR = numParticles;
numMixturetemp = 1000;
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
errorall = zeros(9,n,numStep,numMC);
errornorm = zeros(9,numStep,numMC);
rmse = zeros(9,numStep);
armse = zeros(9,1);

% Chi-square test
chisqall = zeros(9,numStep,numMC);
chinorm = zeros(9,numStep);
achi = zeros(9,1);

chithres = chi2inv(0.9999,n);
chicntall = zeros(9,numStep,numMC);
chicntfin = zeros(9,numStep,1);
chifrac = zeros(9,numStep,1);
chifracavg = zeros(9,1);

% Time-related
elpsdtime = zeros(9,numMC);

% Cluster log
cmeanall = zeros(4,n,numMixturetemp,numStep,numMC);
ccovarall = zeros(4,n,n,numMixturetemp,numStep,numMC);
cweightall = zeros(4,numMixturetemp,numStep,numMC);

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
    error = zeros(9,n,numStep);
    poserror = zeros(9,numStep);
    chisq = zeros(9,numStep); % chi-squared statistic storage
    elpst = zeros(9,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);
    ccovarmc = zeros(4,n,n,numMixturetemp,numStep);
    cweightmc = zeros(4,numMixturetemp,numStep);

    %% For plot
    if figure_view
        fig2 = figure;
        hold on; legend;
        grid
        axis('equal')
        plot(trueState(:,1),trueState(:,2),'--ok','DisplayName','True')
        xlabel('East position (m)')
        ylabel('North position (m)')

        fig3 = figure;
        hold on; legend;
        xlabel('Time (sec)')  
        ylabel('|Position estimation error| (m)')
        xlim([0 141])
        set(gca,'YScale','log')
    end
 
    %% PGM-DS
    if comparison_PGM_DS
        %% Filter parameters
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point
        epsilon = 3;            % distance to determine if a point is a core point 50
        
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
    
        error(1,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(1,1) = sqrt(error(1,1,1)^2+error(1,2,1)^2);
        chisq(1,1) = error(1,:,1)*(temp_particle_var(:,:,1)\error(1,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,ccovar,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(1,:,k,1) = cmean(:,k);
            ccovarmc(1,:,:,k,1) = ccovar(:,:,k);
            cweightmc(1,k,1) = cweight(k,1);
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
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);

            %% For evaluation
            error(1,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(1,i) = sqrt(error(1,1,i)^2+error(1,2,i)^2);
            chisq(1,i) = error(1,:,i)*(temp_particle_var(:,:,i)\error(1,:,i)');
    
            for k = length(cweight)
                cmeanmc(1,:,k,i) = cmean(:,k);
                ccovarmc(1,:,:,k,i) = ccovar(:,:,k);
                cweightmc(1,k,i) = cweight(k,1);
            end   
        end
        
        elpst(1) = toc(tStart1);
        
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'r','DisplayName','PGM-DS')
            figure(fig3); 
            plot(t,poserror(1,:),'r','DisplayName','PGM-DS')
        end
    end

    %% PGM_DU
    if comparison_PGM_DU
        %% Filter parameters
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point
        epsilon = 3;            % distance to determine if a point is a core point 50
        
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
    
        error(2,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(2,1) = sqrt(error(2,1,1)^2+error(2,2,1)^2);
        chisq(2,1) = error(2,:,1)*(temp_particle_var(:,:,1)\error(2,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,ccovar,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(2,:,k,1) = cmean(:,k);
            ccovarmc(2,:,:,k,1) = ccovar(:,:,k);
            cweightmc(2,k,1) = cweight(k,1);
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
            error(2,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(2,i) = sqrt(error(2,1,i)^2+error(2,2,i)^2);
            chisq(2,i) = error(2,:,i)*(temp_particle_var(:,:,i)\error(2,:,i)');
    
            for k = length(cweight)
                cmeanmc(2,:,k,i) = cmean(:,k);
                ccovarmc(2,:,:,k,i) = ccovar(:,:,k);
                cweightmc(2,k,i) = cweight(k,1);
            end  
        end
        
        elpst(2) = toc(tStart2);
        
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'g','DisplayName','PGM-DU')
            figure(fig3); 
            plot(t,poserror(2,:),'g','DisplayName','PGM-DU')
        end
    end

    %% PGM1
    if comparison_PGM1
        %% Filter parameters
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
    
        error(3,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(3,1) = sqrt(error(3,1,1)^2+error(3,2,1)^2);
        chisq(3,1) = error(3,:,1)*(temp_particle_var(:,:,1)\error(3,:,1)');
        
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,ccovar,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(3,:,k,1) = cmean(:,k);
            ccovarmc(3,:,:,k,1) = ccovar(:,:,k);
            cweightmc(3,k,1) = cweight(k,1);
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
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas',likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);

            %% For evaluation
            error(3,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(3,i) = sqrt(error(3,1,i)^2+error(3,2,i)^2);
            chisq(3,i) = error(3,:,i)*(temp_particle_var(:,:,i)\error(3,:,i)');
    
            for k = length(cweight)
                cmeanmc(3,:,k,i) = cmean(:,k);
                ccovarmc(3,:,:,k,i) = ccovar(:,:,k);
                cweightmc(3,k,i) = cweight(k,1);
            end
        end
        
        elpst(3) = toc(tStart3);
    
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'Color','#0072BD','DisplayName','PGM1')
            figure(fig3); 
            plot(t,poserror(3,:),'Color','#0072BD','DisplayName','PGM1')
        end
    end

    %% PGM1_UT
    if comparison_PGM1_UT
        %% Filter parameters
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
    
        error(4,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(4,1) = sqrt(error(4,1,1)^2+error(4,2,1)^2);
        chisq(4,1) = error(4,:,1)*(temp_particle_var(:,:,1)\error(4,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,ccovar,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(4,:,k,1) = cmean(:,k);
            ccovarmc(4,:,:,k,1) = ccovar(:,:,k);
            cweightmc(4,k,1) = cweight(k,1);
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
            error(4,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(4,i) = sqrt(error(4,1,i)^2+error(4,2,i)^2);
            chisq(4,i) = error(4,:,i)*(temp_particle_var(:,:,i)\error(4,:,i)');
    
            for k = length(cweight)
                cmeanmc(4,:,k,i) = cmean(:,k);
                ccovarmc(4,:,:,k,i) = ccovar(:,:,k);
                cweightmc(4,k,i) = cweight(k,1);
            end  
        end
        
        elpst(4) = toc(tStart4);
        
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'Color','#EDB120','DisplayName','PGM1-UT')
            figure(fig3); 
            plot(t,poserror(4,:),'Color','#EDB120','DisplayName','PGM1-UT')
        end
    end

    %% AKKF
    if comparison_AKKF
        %% Filter parameters
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
    
        error(5,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(5,1) = sqrt(error(5,1,1)^2+error(5,2,1)^2);
        chisq(5,1) = error(5,:,1)*(temp_particle_var(:,:,1)\error(5,:,1)');
    
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
            
            error(5,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(5,i) = sqrt(error(5,1,i)^2+error(5,2,i)^2);
            chisq(5,i) = error(5,:,i)*(temp_particle_var(:,:,i)\error(5,:,i)');
        end
        
        elpst(5) = toc(tStart5);
        
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'m','DisplayName','AKKF')
            figure(fig3); 
            plot(t,poserror(5,:),'m','DisplayName','AKKF')
        end
    end

    %% EnKF
    if comparison_EnKF
        %% Initialisation
        ensembles = zeros(n,numEnsembles,numStep);
        ensemble_mean = zeros(n,numStep);
        ensemble_var = zeros(n,n,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
        
        %% First time step  
        ensembles(:,:,1) = initial.*ones(n,numEnsembles)+sqrt(P0)*randn(n,numEnsembles);
        ensemble_var(:,:,1) = P0;
        
        estState(:,1) = mean(ensembles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(6,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(6,1) = sqrt(error(6,1,1)^2+error(6,2,1)^2);
        chisq(6,1) = error(6,:,1)*(temp_particle_var(:,:,1)\error(6,:,1)');
        
        %% Main loop
        tStart6 = tic; 
        
        for i = 2:numStep
            %% prediction
            for e = 1:numEnsembles
                [ensembles(:,e,i),F,G] = ...
                        blindtricyc03_ffunct(ensembles(:,e,i-1),...
                        wbar,i-2,0,ukhist,t,bw);
            end
    
            ensemble_mean(:,i) = mean(ensembles(:,:,i),2);
    
            for j = 1:n
                for k = 1:n
                    if j == k
                        ensemble_var(j,k,i) = var(ensembles(j,:,i));
                    else
                        temp = cov(ensembles(j,:,i),ensembles(k,:,i));
                        ensemble_var(j,k,i) = temp(1,2);
                    end
                end
            end
    
            %% Update
            for e = 1:numEnsembles
                [y,~] = blindtricyc03_haltfunct(ensembles(:,e,i),i-1,0,br,...
                        Xvec,Yvec,rhovec,imykflaghist,meas);
            end
    
            [~,H] = blindtricyc03_haltfunct(ensemble_mean(:,i),i-1,0,br,...
                        Xvec,Yvec,rhovec,imykflaghist,meas);
            K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R);  
    
            for e = 1:numEnsembles
                pert = sqrt(R)*randn(1);
                residual = meas(i,:)'+pert(:,1)-y;
                ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
            end
    
            estState(:,i) = mean(ensembles(:,:,i),2);
    
            %% For evaluation
            temp_particle_var(:,:,i) = ensemble_var(:,:,i);
    
            error(6,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(6,i) = sqrt(error(6,1,i)^2+error(6,2,i)^2);
            chisq(6,i) = error(6,:,i)*(temp_particle_var(:,:,i)\error(6,:,i)');
        end
    
        elpst(6) = toc(tStart6);
    
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'c','DisplayName','EnKF')
            figure(fig3); 
            plot(t,poserror(6,:),'c','DisplayName','EnKF')
        end
    end
    
    %% EKF
    if comparison_EKF
        %% Initialisation
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        x = initial;
        P = P0;
        
        estState(:,1) = x;
        temp_particle_var(:,:,1) = P;
    
        error(7,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(7,1) = sqrt(error(7,1,1)^2+error(7,2,1)^2);
        chisq(7,1) = error(7,:,1)*(temp_particle_var(:,:,1)\error(7,:,1)');
        
        %% Main loop
        tStart7 = tic; 
        
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
            error(7,:,i) = x-trueState(i,:)';
            poserror(7,i) = sqrt(error(7,1,i)^2+error(7,2,i)^2);
            chisq(7,i) = error(7,:,i)*(P\error(7,:,i)');
        end
    
        elpst(7) = toc(tStart7);
    
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'Color','#7E2F8E','DisplayName','EKF');
            figure(fig3); 
            plot(t,poserror(7,:),'Color','#7E2F8E','DisplayName','EKF')
        end
    end
    
    %% UKF
    if comparison_UKF
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
        
        error(8,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(8,1) = sqrt(error(8,1,1)^2+error(8,2,1)^2);
        chisq(8,1) = error(8,:,1)*(temp_particle_var(:,:,1)\error(8,:,1)');
        
        %% Main loop
        tStart8 = tic; 
    
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
            error(8,:,i) = abs(x-trueState(i,:)');
            poserror(8,i) = sqrt(error(8,1,i)^2+error(8,2,i)^2);
            chisq(8,i) = error(8,:,i)*(P\error(8,:,i)'); 
    
            %% Initialisation
            xn = zeros(n,1);
            Pn = zeros(n);
            zn = zeros(nm,1);
            Pzz = zeros(nm);
            Pxz = zeros(n,nm);
        end
        
        elpst(8) = toc(tStart8);
    
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'Color','#4DBEEE','DisplayName','UKF')
            figure(fig3); 
            plot(t,poserror(8,:),'Color','#4DBEEE','DisplayName','UKF')
        end
    end

    %% SIR
    if comparison_SIR
        %% Filter parameters
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
    
        error(9,:,1) = abs(estState(:,1)-trueState(1,:)');
        poserror(9,1) = sqrt(error(9,1,1)^2+error(9,2,1)^2);
        chisq(9,1) = error(9,:,1)*(temp_particle_var(:,:,1)\error(9,:,1)');
        
        %% Main loop
        tStart9 = tic; 
    
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
    
            error(9,:,i) = abs(estState(:,i)-trueState(i,:)');
            poserror(9,i) = sqrt(error(9,1,i)^2+error(9,2,i)^2);
            chisq(9,i) = error(9,:,i)*(temp_particle_var(:,:,i)\error(9,:,i)');
        end
        
        elpst(9) = toc(tStart9);
    
        if figure_view
            figure(fig2); 
            plot(estState(1,:),estState(2,:),'Color',"#77AC30",'DisplayName','SIR')
            figure(fig3); 
            plot(t,poserror(9,:),'Color',"#77AC30",'DisplayName','SIR')
        end
    end
   
    %% End of parfor
    errorall(:,:,:,mc) = error;
    poserrorall(:,:,mc) = poserror;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    cmeanall(:,:,:,:,mc) = cmeanmc;
    ccovarall(:,:,:,:,:,mc) = ccovarmc;
    cweightall(:,:,:,mc) = cweightmc;
end

%% Evaluations
tlim = 141;

% Root means squared error (RMSE)
for fil = 1:9
    for i = 1:numStep
        for mc = 1:numMC
            errornorm(fil,i,mc) = sqrt(errorall(fil,1,i,mc)^2+errorall(fil,2,i,mc)^2); % Position error
        end
    end
end

for fil = 1:9
    for i = 1:numStep
        rmse(fil,i) = sqrt(sum(errornorm(fil,i,1:numMC).^2)/numMC);
    end
end

% Averaged RMSE
for fil = 1:9
    armse(fil) = sum(rmse(fil,:))/numStep;
end

% Plot RMSE
figure; hold on; legend;

set(gca,'YScale','log')

plot(t,rmse(1,:),'r','DisplayName','PGM-DS')
plot(t,rmse(2,:),'g','DisplayName','PGM-DU')
plot(t,rmse(3,:),'Color','#0072BD','DisplayName','PGM1')
plot(t,rmse(4,:),'Color','#EDB120','DisplayName','PGM1-UT')
plot(t,rmse(5,:),'m','DisplayName','AKKF')
% plot(t,rmse(6,:),'c','DisplayName','EnKF')
plot(t,rmse(7,:),'Color','#7E2F8E','DisplayName','EKF')
plot(t,rmse(8,:),'Color','#4DBEEE','DisplayName','UKF')
plot(t,rmse(9,:),'Color','#77AC30','DisplayName','SIR')

xlim([0 tlim])
% ylim([-25 25])
xlabel('Time (sec)')
ylabel('RMSE')
hold off;

% Average terminate position error
for fil = 1:9
    lastpos = poserrorall(fil,end,:); % Position error
    lastpos = reshape(lastpos,[1,numMC]);

    % atermerr(fil) = mean(lastpos);
    atermerr(fil) = sqrt(sum(lastpos.^2)/numMC);
end

atermerr = atermerr';

% Chi-squared
for fil = 1:9
    for i = 1:numStep
        chinorm(fil,i) = sum(chisqall(fil,i,:))/(n*numMC);
    end
end

for fil = 1:9
    achi(fil) = sum(chinorm(fil,:))/numStep;
end

% Plot Chisq
figure; hold on; legend;
plot(t,chinorm(1,:),'r','DisplayName','PGM-DS')
plot(t,chinorm(2,:),'g','DisplayName','PGM-DU')
plot(t,chinorm(3,:),'Color','#0072BD','DisplayName','PGM1')
plot(t,chinorm(4,:),'Color','#EDB120','DisplayName','PGM1-UT')
plot(t,chinorm(5,:),'m','DisplayName','AKKF')
% plot(t,chinorm(6,:),'c','DisplayName','EnKF')
plot(t,chinorm(7,:),'Color','#7E2F8E','DisplayName','EKF')
plot(t,chinorm(8,:),'Color','#4DBEEE','DisplayName','UKF')
plot(t,chinorm(9,:),'Color','#77AC30','DisplayName','SIR')

xlim([0 tlim])
% ylim([-25 25])
xlabel('Time (sec)')
ylabel('State-Error Chi-squared statistic/nx')
hold off;

% Counting Chi-squared
for fil = 1:9
    for i = 1:numStep
        for mc = 1:numMC
            if chisqall(fil,i,mc) > chithres
                chicntall(fil,i,mc) = 1;
            end
        end
    end
end

for fil = 1:9
    for i = 1:numStep
        chicntfin(fil,i) = sum(chicntall(fil,i,:) == 1);
        chifrac(fil,i) = chicntfin(fil,i)/numMC;
    end
end

for fil = 1:9
    chifracavg(fil) = mean(chifrac(fil,:),2);
end

% Plot Chisq
figure; hold on; legend;
plot(t,chifrac(1,:),'r','DisplayName','PGM-DS')
plot(t,chifrac(2,:),'g','DisplayName','PGM-DU')
plot(t,chifrac(3,:),'Color','#0072BD','DisplayName','PGM1')
plot(t,chifrac(4,:),'Color','#EDB120','DisplayName','PGM1-UT')
plot(t,chifrac(5,:),'m','DisplayName','AKKF')
% plot(t,chifrac(6,:),'c','DisplayName','EnKF')
plot(t,chifrac(7,:),'Color','#7E2F8E','DisplayName','EKF')
plot(t,chifrac(8,:),'Color','#4DBEEE','DisplayName','UKF')
plot(t,chifrac(9,:),'Color','#77AC30','DisplayName','SIR')

xlim([0 tlim])
% ylim([-25 25])
xlabel('Time (sec)')
ylabel('Fraction of Cases')
hold off;

elpadtavg = mean(elpsdtime,2);

%% Save variables
save('4_Blind_Tricyclist_MC')
