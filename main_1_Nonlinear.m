%% Particle Gaussian Mixture Filter with DBSCAN (and with UT)
% Sukkeun Kim (Sukkeun.Kim@cranfield.ac.uk)

clc; clear all; close all;
addpath('functions')

%% Simulation Settings
figure_view = 1;
comparison_PGM_DU = 1;      % 1: compare with PGM_DU  / 0: no comparison
comparison_PGM_DS = 1;      % 1: compare with PGM_DS  / 0: no comparison
comparison_PGM_UT = 1;      % 1: compare with PGM_UT  / 0: no comparison
comparison_PGM = 1;         % 1: compare with PGM  / 0: no comparison
comparison_AKKF = 1;        % 1: compare with AKKF  / 0: no comparison
comparison_EnKF = 0;        % 1: compare with EnKF / 0: no comparison
comparison_EKF = 0;         % 1: compare with EKF / 0: no comparison
comparison_UKF = 1;         % 1: compare with UKF / 0: no comparison
comparison_PF = 1;          % 1: compare with PF / 0: no comparison
gaussian_update = 1;        % 1: Gaussian distribution / 0: quantisation operator

numParticles = 50; %50
numEnsembles = numParticles;
numMixturetemp = 20;
numMC = 1;

rng(42);

%% Define a Scenario
% Time
dt = 1;
tf = 52;
t = 0:dt:tf;
numStep = length(t);

% Parameters
n = 1; % dimension of the state
nm = 1; % dimension of the measurement 

Q = 10.0;
R = 1.0;
Rmat = R*eye(nm);
merging_thres = chi2inv(0.995,n); % threshold for merging mixtures

trueinitial = 0.0;
P0 = 2.0;

initial = trueinitial;

% System/measurement equations
funsys = @(x,w,k) 0.5*x(1,:)+25*x(1,:)/(1+x(1,:).^2)+8*cos(1.2*k)+w;
funmeas = @(x,v) (diag(x(1,:)'*x(1,:)))'/20+v;
jacobianF = @(x) 0.5+(25*(1+x(1,:)*x(1,:))-50*x(1,:)*x(1,:))/(1+x(1,:)*x(1,:))*(1+x(1,:)*x(1,:));
jacobianH = @(x) x(1,:)/10;
G = eye(n);

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

looptic = tic;

%% Monte Carlo simulation
for mc = 1:numMC 
    disptemp = ['Monete Carlo iteration: ', num2str(mc)];
    disp(disptemp)

    % Generate true states
    trueState = zeros(n,numStep);
    procNoise = sqrt(Q)*randn(n,numStep);
    trueState(:,1) = trueinitial+sqrt(P0)*randn(n,1);

    for i = 2:numStep
        trueState(:,i) = funsys(trueState(:,i-1),procNoise(i-1),i-1);
    end
    
    % Generate measurements
    measNoise = sqrt(R)*randn(nm,numStep);
    meas = funmeas(trueState,measNoise);
    
    % For evaluation
    error = zeros(9,n,numStep);
    chisq = zeros(9,numStep); % chi-squared statistic storage
    elpst = zeros(9,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);
    ccovarmc = zeros(4,n,n,numMixturetemp,numStep);
    cweightmc = zeros(4,numMixturetemp,numStep);

    %% For plot
    if figure_view
        fig2 = figure; 
        hold on; legend;
        plot(t,trueState(1,:),'--ok','DisplayName', 'True')
        xlim([0 tf])
        xlabel('Time (sec)')
        ylabel('Estimated state')

        fig3 = figure;
        hold on; legend;
        xlim([0 tf])
        xlabel('Time (sec)')
        ylabel('|Estimation error|')
    end

    %% PGM-DU
    if comparison_PGM_DU
    %% Filter Parameters
    numMixture_max = n+1; % number of Gaussian mixtures
    minpts = n+1; % Minimum number of neighbors for a core point n + 1
    epsilon = 10; % Distance to determine if a point is a core point 3

    numSigma = 2*n+1;
    alpha = 1.3;
    beta = 1.5;
    kappa = 0;
    lambda = 0.2;
    % lambda = alpha^2*(n+kappa)-n;
    
    %% initialise
    particle_state = zeros(n,numParticles,numStep);
    particle_mean = zeros(n,numMixture_max,numStep);
    particle_var = zeros(n,n,numMixture_max,numStep);
    particle_meas = zeros(nm,numParticles,numStep);

    wm = zeros(numSigma,1);
    wc = zeros(numSigma,1);

    estState = zeros(n,numStep);
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    particle_state(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);

    for p = 1:numParticles
        particle_meas(:,p,1) = funmeas(particle_state(:,p,1),0);
    end 

    wm(1,:) = lambda/(n+lambda);
    wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);

    for j = 2:numSigma   
        wm(j,:) = 1/(2*(n+lambda));
        wc(j,:) = wm(j);
    end
    
    estState(:,1) = mean(particle_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(1,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(1,1) = error(1,:,1)*(temp_particle_var(:,:,1)\error(1,:,1)');

    % Clustering
    [~,particle_mean,particle_var,~,...
            cmean,ccovar,cweight,~,~] = ...
               PGM_DS_clustering(1,n,numParticles,...
                              particle_state,particle_mean,particle_var,...
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
            particle_state(:,p,i) = funsys(particle_state(:,p,i-1),sqrt(Q)*randn(1),i-1);
        end
    
        %% Clustering
        [particle_clust,particle_mean,particle_var,numMixture,...
                    ~,~,cweight,likelih,idx] = ...
                       PGM_DS_clustering(i,n,numParticles,...
                                      particle_state,particle_mean,particle_var,...
                                      epsilon,minpts,numMixture_max,1,0);

        %% Merging
        [~,particle_clust,particle_mean,particle_var,...
            numMixture,cweight,idx] = ...
                       PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                    particle_clust,particle_state,...
                                    particle_mean,particle_var);

        %% Update
        [particle_state,estState,temp_particle_var,...
         cmean,ccovar,cweight] = ...
                       PGM_DS_update(i,n,nm,numStep,numParticles,...
                                   numMixture,numMixture_max,...
                                   1,cweight,idx,meas,likelih,R,...
                                   particle_state,particle_clust,...
                                   particle_meas,particle_mean,...
                                   particle_var,estState,...
                                   temp_particle_var,...
                                   1,numSigma,lambda,wc,wm,funmeas);

        %% For evaluation
        error(1,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(1,i) = error(2,:,i)*(temp_particle_var(:,:,i)\error(2,:,i)');

        for k = length(cweight)
            cmeanmc(1,:,k,i) = cmean(:,k);
            ccovarmc(1,:,:,k,i) = ccovar(:,:,k);
            cweightmc(1,k,i) = cweight(k,1);
        end  
    end
    
    elpst(1) = toc(tStart1);

    temperror = zeros(1,length(error(1,1,:)));
    temperror(:) = error(1,1,:);
    
    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'r','DisplayName', 'PGM-DU')
        figure(fig3); 
        plot(t,temperror,'r','DisplayName', 'PGM-DU')
    end
    end

    %% PGM-DS
    if comparison_PGM_DS
    %% Filter Parameters
    numMixture_max = n+1; % number of Gaussian mixtures
    minpts = n+1; % Minimum number of neighbors for a core point n + 1
    epsilon = 10; % Distance to determine if a point is a core point 3
    
    %% Initialise
    particle_state = zeros(n,numParticles,numStep);
    particle_mean = zeros(n,numMixture_max,numStep);
    particle_var = zeros(n,n,numMixture_max,numStep);
    particle_meas = zeros(nm,numParticles,numStep);

    estState = zeros(n,numStep);
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    particle_state(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);

    for p = 1:numParticles
        particle_meas(:,p,1) = funmeas(particle_state(:,p,1),0);
    end 

    estState(:,1) = mean(particle_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(2,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(2,1) = error(2,:,1)*(temp_particle_var(:,:,1)\error(2,:,1)');

    % Clustering
    [~,particle_mean,particle_var,~,...
            cmean,ccovar,cweight,~,~] = ...
               PGM_DS_clustering(1,n,numParticles,...
                              particle_state,particle_mean,particle_var,...
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
            particle_state(:,p,i) = funsys(particle_state(:,p,i-1),sqrt(Q)*randn(1),i-1);
        end
    
        %% Clustering
        [particle_clust,particle_mean,particle_var,numMixture,...
                    ~,~,cweight,likelih,idx] = ...
                       PGM_DS_clustering(i,n,numParticles,...
                                      particle_state,particle_mean,particle_var,...
                                      epsilon,minpts,numMixture_max,1,0);

        %% Merging
        [~,particle_clust,particle_mean,particle_var,...
            numMixture,cweight,idx] = ...
                       PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                    particle_clust,particle_state,...
                                    particle_mean,particle_var);

        %%  Update
        for p = 1:numParticles
            particle_meas(:,p,i) = funmeas(particle_state(:,p,i),sqrt(R)*randn(nm,1));
        end

        [particle_state,estState,temp_particle_var,...
         cmean,ccovar,cweight] = ...
                       PGM_DS_update(i,n,nm,numStep,numParticles,...
                                   numMixture,numMixture_max,...
                                   1,cweight,idx,meas,likelih,R,...
                                   particle_state,particle_clust,...
                                   particle_meas,particle_mean,...
                                   particle_var,estState,...
                                   temp_particle_var,0,0,0,0,0,0);

        %% For evaluation
        error(2,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(2,i) = error(2,:,i)*(temp_particle_var(:,:,i)\error(2,:,i)');

        for k = length(cweight)
            cmeanmc(2,:,k,i) = cmean(:,k);
            ccovarmc(2,:,:,k,i) = ccovar(:,:,k);
            cweightmc(2,k,i) = cweight(k,1);
        end
    end
    
    elpst(2) = toc(tStart2);

    temperror = zeros(1,length(error(2,1,:)));
    temperror(:) = error(2,1,:);
    
    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'g','DisplayName', 'PGM-DS')
        figure(fig3);
        plot(t,temperror,'g','DisplayName', 'PGM-DS')
    end
    end

    %% PGM-UT
    if comparison_PGM_UT
    %% Filter Parameters
    numMixture_max = 3; % number of Gaussian mixtures

    numSigma = 2*n+1;
    alpha = 1.3;
    beta = 1.5;
    kappa = 0;
    lambda = 0.2;
    % lambda = alpha^2*(n+kappa)-n;
    
    %% initialise
    particle_state = zeros(n,numParticles,numStep);
    particle_mean = zeros(n,numMixture_max,numStep);
    particle_var = zeros(n,n,numMixture_max,numStep);
    particle_meas = zeros(nm,numParticles,numStep);

    wm = zeros(numSigma,1);
    wc = zeros(numSigma,1);

    estState = zeros(n,numStep);
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    particle_state(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);

    for p = 1:numParticles
        particle_meas(:,p,1) = funmeas(particle_state(:,p,1),0);
    end 

    wm(1,:) = lambda/(n+lambda);
    wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);

    for j = 2:numSigma   
        wm(j,:) = 1/(2*(n+lambda));
        wc(j,:) = wm(j);
    end
    
    estState(:,1) = mean(particle_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(3,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(3,1) = error(3,:,1)*(temp_particle_var(:,:,1)\error(3,:,1)');

    % Clustering
    [~,particle_mean,particle_var,~,...
                cmean,ccovar,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particle_state,particle_mean,particle_var,...
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
            particle_state(:,p,i) = funsys(particle_state(:,p,i-1),sqrt(Q)*randn(1),i-1);
        end
    
        %% Clustering
        [particle_clust,particle_mean,particle_var,numMixture,...
                    ~,~,cweight,likelih,idx] = ...
                       PGM_DS_clustering(i,n,numParticles,...
                                      particle_state,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,0);

        %% Merging
        [~,particle_clust,particle_mean,particle_var,...
            numMixture,cweight,idx] = ...
                       PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                    particle_clust,particle_state,...
                                    particle_mean,particle_var);

        %% Update
        [particle_state,estState,temp_particle_var,...
         cmean,ccovar,cweight] = ...
                       PGM_DS_update(i,n,nm,numStep,numParticles,...
                                   numMixture,numMixture_max,...
                                   1,cweight,idx,meas,likelih,R,...
                                   particle_state,particle_clust,...
                                   particle_meas,particle_mean,...
                                   particle_var,estState,...
                                   temp_particle_var,...
                                   1,numSigma,lambda,wc,wm,funmeas);

        %% For evaluation
        error(3,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(3,i) = error(3,:,i)*(temp_particle_var(:,:,i)\error(3,:,i)');

        for k = length(cweight)
            cmeanmc(3,:,k,i) = cmean(:,k);
            ccovarmc(3,:,:,k,i) = ccovar(:,:,k);
            cweightmc(3,k,i) = cweight(k,1);
        end  
    end
    
    elpst(3) = toc(tStart3);

    temperror = zeros(1,length(error(3,1,:)));
    temperror(:) = error(3,1,:);
    
    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'Color','#0072BD','DisplayName', 'PGM-UT')
        figure(fig3); 
        plot(t,temperror,'Color','#0072BD','DisplayName', 'PGM-UT')
    end
    end
  
    %% PGM
    if comparison_PGM
    %% Filter Parameters
    numMixture_max = 3; % number of Gaussian mixtures
    
    %% Initialise
    particle_state = zeros(n,numParticles,numStep);
    particle_mean = zeros(n,numMixture_max,numStep);
    particle_var = zeros(n,n,numMixture_max,numStep);
    particle_meas = zeros(nm,numParticles,numStep);

    estState = zeros(n,numStep);
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    particle_state(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);

    for p = 1:numParticles
        particle_meas(:,p,1) = funmeas(particle_state(:,p,1),0);
    end 

    estState(:,1) = mean(particle_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(4,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(4,1) = error(4,:,1)*(temp_particle_var(:,:,1)\error(4,:,1)');

    % Clustering
    [~,particle_mean,particle_var,~,...
                cmean,ccovar,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particle_state,particle_mean,particle_var,...
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
            particle_state(:,p,i) = funsys(particle_state(:,p,i-1),sqrt(Q)*randn(1),i-1);
        end
    
        %% Clustering
        [particle_clust,particle_mean,particle_var,numMixture,...
                    ~,~,cweight,likelih,idx] = ...
                       PGM_DS_clustering(i,n,numParticles,...
                                      particle_state,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,0);

        %% Merging
        [~,particle_clust,particle_mean,particle_var,...
            numMixture,cweight,idx] = ...
                       PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                    particle_clust,particle_state,...
                                    particle_mean,particle_var);

        %% Update
        for p = 1:numParticles
            particle_meas(:,p,i) = funmeas(particle_state(:,p,i),sqrt(R)*randn(nm,1));
        end
    
        [particle_state,estState,temp_particle_var,...
         cmean,ccovar,cweight] = ...
                       PGM_DS_update(i,n,nm,numStep,numParticles,...
                                   numMixture,numMixture_max,...
                                   1,cweight,idx,meas,likelih,R,...
                                   particle_state,particle_clust,...
                                   particle_meas,particle_mean,...
                                   particle_var,estState,...
                                   temp_particle_var,0,0,0,0,0,0);

        %% For evaluation
        error(4,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(4,i) = error(4,:,i)*(temp_particle_var(:,:,i)\error(4,:,i)');

        for k = length(cweight)
            cmeanmc(4,:,k,i) = cmean(:,k);
            ccovarmc(4,:,:,k,i) = ccovar(:,:,k);
            cweightmc(4,k,i) = cweight(k,1);
        end  
    end
    
    elpst(4) = toc(tStart4);

    temperror = zeros(1,length(error(4,1,:)));
    temperror(:) = error(4,1,:);
    
    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'Color','#EDB120','DisplayName', 'PGM')
        figure(fig3);
        plot(t,temperror,'Color','#EDB120','DisplayName', 'PGM')
    end
    end

    %% AKKF
    if comparison_AKKF
    %% Filter Parameters
    AKKF.kernel = 1;  % "Quadratic" "Quartic" "Gaussian"
    AKKF.N_P = numParticles;

    AKKF.c = 1; %
    AKKF.lambda = 10^(-3);  % kernel normisation parameter
    AKKF.lambda_KKF = 10^(-3);  % kernel normisation parameter for Kernel Kalman filter gain calculation
    AKKF.poly_para_b = 10^(3); % polynomial triack scale for the bearing tracking
    AKKF.Var_Gaussian = R;

    %% Initialise
    estState = zeros(n,numStep); % storage for state estimates
    temp_particle_var = zeros(n,n,numStep);
    
    %% First time step
    AKKF.X_P(:,:,1) = initial+sqrt(P0)*randn(1,numParticles);

    % Data space
%     AKKF.X_P(:,:,1) = mgd(AKKF.N_P, 4, Tar.X(:,1), Sys.P0).'; % state particles
    
    % Kernel space
    AKKF.W_minus(:,1) = ones(AKKF.N_P, 1)/AKKF.N_P;
    AKKF.S_minus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
    
    AKKF.W_plus(:,1) = ones(AKKF.N_P, 1)/AKKF.N_P;
    AKKF.S_plus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
    
    AKKF.X_P_proposal(:,:,1) =  AKKF.X_P(:,:,1);
    
    AKKF.X_est(:,1) = AKKF.X_P(:,:,1) * AKKF.W_plus(:,1); %state mean
    AKKF.X_C(:,:,1) =   AKKF.X_P(:,:,1) * AKKF.S_plus(:,:,1) * AKKF.X_P(:,:,1).'; %state covariance
    
    for p = 1:numParticles
        AKKF.Z_P(:,p,1) = funmeas(AKKF.X_P(:,p,1),0);
    end 

    estState(:,1) = mean(AKKF.X_P(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(5,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(5,1) = error(5,:,1)*(temp_particle_var(:,:,1)\error(5,:,1)');

    %% Main loop
    tStart5 = tic; 

    for i = 2:numStep
        %% Prediction
        for p = 1:numParticles
             AKKF.X_P(:,p,i) = funsys(AKKF.X_P_proposal(:,p,i-1),sqrt(Q)*randn(1),i-1);
        end

        [AKKF] = AKKF_predict(AKKF, i);

        %% Update
        for p = 1:numParticles
            AKKF.Z_P(:,p,i) = funmeas(AKKF.X_P(:,p,i),sqrt(R)*randn(nm,1));
        end

        [AKKF] = AKKF_update(meas, AKKF, R, i);
        [AKKF] = AKKF_proposal(AKKF, i);

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
        
        error(5,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(5,i) = error(5,:,i)*(temp_particle_var(:,:,i)\error(5,:,i)');
    end
    
    elpst(5) = toc(tStart5);

    temperror = zeros(1,length(error(5,1,:)));
    temperror(:) = error(5,1,:);
        
    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'m','DisplayName', 'AKKF')
        figure(fig3);
        plot(t,temperror,'m','DisplayName', 'AKKF')
    end
    end

    %% EnKF
    if comparison_EnKF
    %% Initialise
    ensemble_state = zeros(n,numEnsembles,numStep);
    ensemble_mean = zeros(n,numStep);
    ensemble_var = zeros(n,n,numStep);

    estState = zeros(n,numStep); % storage for state estimates
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    ensemble_state(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
    ensemble_var(:,:,1) = P0;
    
    estState(:,1) = mean(ensemble_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(6,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(6,1) = error(6,:,1)*(temp_particle_var(:,:,1)\error(6,:,1)');
    
    %% Main loop
    tStart6 = tic; 
    
    for i = 2:numStep
        %% Prediction
        for e = 1:numParticles
            ensemble_state(:,e,i) = funsys(ensemble_state(:,e,i-1),sqrt(Q)*randn(1),i-1);
        end

        ensemble_mean(:,i) = mean(ensemble_state(:,:,i),2);

        for j = 1:n
            for k = 1:n
                if j == k
                    ensemble_var(j,k,i) = var(ensemble_state(j,:,i));
                else
                    temp = cov(ensemble_state(j,:,i),ensemble_state(k,:,i));
                    ensemble_var(j,k,i) = temp(1,2);
                end
            end
        end

        %% Update
        H = jacobianH(ensemble_mean(:,i));
        K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+Rmat);   

        for e = 1:numEnsembles
            pert = sqrt(R)*randn(1);
            residual = meas(:,i)+pert-funmeas(ensemble_state(:,e,i),0);
            ensemble_state(:,e,i) = ensemble_state(:,e,i)+K*residual;
        end

        estState(:,i) = mean(ensemble_state(:,:,i),2);

        %% For evaluation
        temp_particle_var(:,:,i) = ensemble_var(:,:,i);

        error(6,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(6,i) = error(6,:,i)*(temp_particle_var(:,:,i)\error(6,:,i)');
    end

    elpst(6) = toc(tStart6);

    temperror = zeros(1,length(error(6,1,:)));
    temperror(:) = error(6,1,:);

    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'c','DisplayName', 'EnKF')
        figure(fig3);
        plot(t,temperror,'c','DisplayName', 'EnKF')
    end
    end

    %% EKF
    if comparison_EKF
    %% Initialise
    estState = zeros(n,numStep);
    temp_particle_var = zeros(n,n,numStep);
    
    %% First time step
    x = initial;
    P = P0;

    estState(:,1) = x;
    temp_particle_var(:,:,1) = P;

    error(7,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(7,1) = error(7,:,1)*(temp_particle_var(:,:,1)\error(7,:,1)');
    
    %% Main loop
    tStart7 = tic; 

    for i = 2:numStep
        %% Prediction
        x = funsys(x,0,i-1);
        F = jacobianF(x);
        P = F*P*F'+ Q; 

        %% Update
        residual = meas(:,i) - funmeas(x,0);
        H = jacobianH(x);
        K = P*H'/(H*P*H'+R);   
        x = x+K*residual;
        P = (eye(1)-K*H)*P;    
        estState(:,i) = x;

        %% For evaluation
        error(7,:,i) = x-trueState(:,i);
        chisq(7,i) = error(7,:,i)*(P\error(7,:,i)');   
    end
    
    elpst(7) = toc(tStart7);

    temperror = zeros(1,length(error(7,1,:)));
    temperror(:) = error(7,1,:);

    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'Color','#7E2F8E','DisplayName', 'EKF')
        figure(fig3);
        plot(t,temperror,'Color','#7E2F8E','DisplayName', 'EKF')
    end
    end
    
    %% UKF
    if comparison_UKF
    %% Filter Parameters
    numSigma = 2*n+1;
    alpha = 1.3;
    beta = 1.5;
    kappa = 0;
    lambda = 0.2;
    % lambda = alpha^2*(n+kappa)-n;
    
    %% Initialise
    xs = zeros(n,numSigma,numStep);
    zs = zeros(nm,numSigma,numStep);
    wm = zeros(numSigma);
    wc = zeros(numSigma);
    xn = zeros(n,1);
    Pn = zeros(n);
    zn = zeros(nm,1);
    Pz = zeros(nm);
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

    error(8,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(8,1) = error(8,:,1)*(temp_particle_var(:,:,1)\error(8,:,1)');
   
    %% Main loop
    tStart8 = tic; 

    for i = 2:numStep
        %% Sigma points
        sqrt_P = chol((n+lambda)*P, 'lower'); % Square root of scaled covariance
        xs(:,1,i) = x;
        for j = 1:n
            xs(:,j+1,i) = x+sqrt_P(:,j);
            xs(:,j+n+1,i) = x-sqrt_P(:,j);
        end

        %% Prediction
        for j = 1:numSigma
            xs(:,j,i) = funsys(xs(:,j,i),0,i-1);
            xn = xn + wm(j)*xs(:,j,i);
        end
        
        for j = 1:numSigma
            Pn = Pn + wc(j)*(xs(:,j,i)-xn)*(xs(:,j,i)-xn)';
        end
        Pn = Pn + G*Q*G';
    
        %% Update
        for j = 1:numSigma
    	    zs(:,j,i) = funmeas(xs(:,j,i),0);
            zn = zn + wm(j)*zs(:,j,i);
        end

        for j = 1:numSigma
            Pz = Pz + wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
        end
        Pz = Pz + Rmat;
    
        for j = 1:numSigma
            Pxz = Pxz + wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
        end
    
        K = Pxz/Pz;
        x = xn + K*(meas(:,i)-zn);
        P = Pn -K*Pz*K';

        %% Covariance correction tricks
        if all(eig(P) > 0)
            
        else
            while all(eig(P) < 0)
                P = (P+P.')/2;
                P = P+0.1*eye(n);
            end
        end
    
        estState(:,i) = x;

        %% For evaluation
        error(8,:,i) = abs(x-trueState(:,i));
        chisq(8,i) = error(8,:,i)*(P\error(8,:,i)');    

        %% Initialisation
        xn = zeros(n,1);
        Pn = zeros(n);
        zn = zeros(nm,1);
        Pz = zeros(nm);
        Pxz = zeros(n,nm);
    end
    
    elpst(8) = toc(tStart8);

    temperror = zeros(1,length(error(8,1,:)));
    temperror(:) = error(8,1,:);

    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'Color','#4DBEEE','DisplayName', 'UKF')
        figure(fig3);
        plot(t,temperror,'Color','#4DBEEE','DisplayName', 'UKF')
    end
    end
    
    %% PF
    if comparison_PF
    %% Filter parameters
    numEffparticles = [];

    %% Initialise
    particle_state = zeros(n,numParticles,numStep);
    particle_w = zeros(numParticles,numStep);

    estState = zeros(n,numStep); % storage for state estimates
    temp_particle_var = zeros(n,n,numStep);

    %% First time step
    particle_state(:,:,1) = initial+sqrt(P0)*randn(1,numParticles);
    particle_w(:,1) = 1/numParticles*ones(numParticles,1);

    estState(:,1) = mean(particle_state(:,:,1),2);
    temp_particle_var(:,:,1) = P0;

    error(9,:,1) = abs(estState(:,1)-trueState(:,1));
    chisq(9,1) = error(9,:,1)*(temp_particle_var(:,:,1)\error(9,:,1)');
    
    %% Main loop
    tStart9 = tic;   

    for i = 2:numStep
        for p = 1:numParticles
            %% Prediction
            particle_state(:,p,i) = funsys(particle_state(:,p,i-1),sqrt(Q)*randn(1),i-1);
            
            %% Update
            residual = meas(:,i)-funmeas(particle_state(:,p,i),0);
            particle_w(p,i) = 1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
        end

        %% Resample
        if all(particle_w(:,i) == 0)
            particle_w(:,i) = 1/numParticles;
        end

        particle_w(:,i) = particle_w(:,i)/sum(particle_w(:,i));
        particle_state(:,:,i) = resample_eff(particle_w(:,i),...
                                particle_state(:,:,i),numEffparticles);
        
        estState(:,i) = mean(particle_state(:,:,i),2);

        %% For evaluation
        for j = 1:n
            for k = 1:n
                if j == k
                    temp_particle_var(j,k,i) = var(particle_state(j,:,i));
                else
                    temp = cov(particle_state(j,:,i),particle_state(k,:,i));
                    temp_particle_var(j,k,i) = temp(1,2);
                end
            end
        end

        error(9,:,i) = abs(estState(:,i)-trueState(:,i));
        chisq(9,i) = error(9,:,i)*(temp_particle_var(:,:,i)\error(9,:,i)');
    end
    
    elpst(9) = toc(tStart9);

    temperror = zeros(1,length(error(9,1,:)));
    temperror(:) = error(9,1,:);

    if figure_view
        figure(fig2); 
        plot(t,estState(1,:),'Color',"#77AC30",'DisplayName', 'PF')
        figure(fig3);
        plot(t,temperror,'Color',"#77AC30",'DisplayName', 'PF')
    end
    end  
    %% End of parfor
    errorall(:,:,:,mc) = error;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    cmeanall(:,:,:,:,mc) = cmeanmc;
    ccovarall(:,:,:,:,:,mc) = ccovarmc;
    cweightall(:,:,:,mc) = cweightmc;
end
looptoc = toc(looptic);

%% Evaluations
% Root means squared error (RMSE)
for fil = 1:9
    for i = 1:numStep
        for mc = 1:numMC
            errornorm(fil,i,mc) = vecnorm(errorall(fil,:,i,mc));
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
    armse(fil) = sum(rmse(fil,:))/numStep;  % PGM-UT/PGM/UKF/PF
end

% Plot RMSE
figure; hold on; legend;
plot(t,rmse(1,:),'r','DisplayName', 'PGM-DU')
plot(t,rmse(2,:),'g','DisplayName', 'PGM-DS')
plot(t,rmse(3,:),'Color','#0072BD','DisplayName', 'PGM-UT')
plot(t,rmse(4,:),'Color','#EDB120','DisplayName', 'PGM')
plot(t,rmse(5,:),'m','DisplayName', 'AKKF')
% plot(t,rmse(6,:),'c','DisplayName', 'EnKF')
% plot(t,rmse(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,rmse(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,rmse(9,:),'Color','#77AC30','DisplayName', 'PF')

xlim([0 tf])
% ylim([0 15])
xlabel('Time (sec)')
ylabel('RMSE')
hold off;

% Chi-Square
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
plot(t,chinorm(1,:),'r','DisplayName', 'PGM-DU')
plot(t,chinorm(2,:),'g','DisplayName', 'PGM-DS')
plot(t,chinorm(3,:),'Color','#0072BD','DisplayName', 'PGM-UT')
plot(t,chinorm(4,:),'Color','#EDB120','DisplayName', 'PGM')
plot(t,chinorm(5,:),'m','DisplayName', 'AKKF')
% plot(t,chinorm(6,:),'c','DisplayName', 'EnKF')
% plot(t,chinorm(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,chinorm(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,chinorm(9,:),'Color','#77AC30','DisplayName', 'PF')

xlim([0 tf])
ylim([0 30])
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
plot(t,chifrac(1,:),'r','DisplayName', 'PGM-DU')
plot(t,chifrac(2,:),'g','DisplayName', 'PGM-DS')
plot(t,chifrac(3,:),'Color','#0072BD','DisplayName', 'PGM-UT')
plot(t,chifrac(4,:),'Color','#EDB120','DisplayName', 'PGM')
plot(t,chifrac(5,:),'m','DisplayName', 'AKKF')
% plot(t,chifrac(6,:),'c','DisplayName', 'EnKF')
% plot(t,chifrac(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,chifrac(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,chifrac(9,:),'Color','#77AC30','DisplayName', 'PF')

xlim([0 tf])
% ylim([-25 25])
xlabel('Time (sec)')
ylabel('Fraction of Cases')
hold off;

elpadtavg = mean(elpsdtime,2);

%% Save variables
save('1_Nonlinear_MC')
