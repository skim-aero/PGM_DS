%% Particle Gaussian Mixture Filter with DBSCAN (and with UT)
% Sukkeun Kim (Sukkeun.Kim@cranfield.ac.uk)

clc; clear; close all;
addpath(genpath('functions'));

warning('off','all')
warning

%% Simulation settings
figure_view = 1;
animation_view = 0;         % 1: show pdf animation / 0: no pdf animation
recordVid_view = 0;         % 1: record pdf animation / 0: no record
comparison_PGM_DS = 1;      % 1: compare with PGM_DS  / 0: no comparison
comparison_PGM_DU = 1;      % 1: compare with PGM_DU  / 0: no comparison
comparison_PGM1 = 1;        % 1: compare with PGM1  / 0: no comparison
comparison_PGM1_UT = 1;     % 1: compare with PGM1_UT  / 0: no comparison
comparison_AKKF = 1;        % 1: compare with AKKF / 0: no comparison
comparison_EnKF = 0;        % 1: compare with EnKF / 0: no comparison
comparison_EKF = 0;         % 1: compare with EKF / 0: no comparison
comparison_UKF = 1;         % 1: compare with UKF / 0: no comparison
comparison_SIR = 1;         % 1: compare with SIR / 0: no comparison

numParticles = 100; % 100
numEnsembles = numParticles;
numMixturetemp = 10;
numMC = 1;

rng(22); % 42 22

%% Define a Scenario
% Time
dt = 0.05;
tf = 10;
t = 0:dt:tf;
numStep = length(t);

theta = 0:0.1:2*pi;
framerate = fix(numStep/tf);

% Parameters
n = 2;      % dimension of the state
nm = 1;     % dimension of the measurement 

L = 3;
g = 9.81;

Q = 10;
R = 0.02; % 0.02
merging_thres = chi2inv(0.995,n); % threshold for merging mixtures

trueinitial = [0.4*pi;0];
P0 = 0.4*eye(n); % 0.4*eye(n)

initial = [0;0]; % initial guess

% UT parameters
numSigma = 2*n+1;
alpha = 1;
beta = 2;
kappa = 0;
lambda = alpha^2*(n+kappa)-n;

% System/measurement equations
funsys = @(x,w,k) [wrapToPi(mod(x(1,:)+dt*x(2,:)+dt^2*(-g+w)*sin(x(1,:))/L,2*pi));
                x(2,:)+dt*(-g+w)*sin(x(1,:))/L];
funmeas = @(x,v) abs(L*sin(x(1,:)))+v;
funpos = @(x) [L*sin(x(1,:));-L*cos(x(1,:))];
jacobianH = @(x) [L*cos(x(1)) 0];
F = [1 dt;0 1];
G = [dt^2/2;dt]/L;

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
correctcnt = zeros(9,numMC);
cnttotal = zeros(9,1);
divergecnt = zeros(1,numMC);

% Cluster log
cmeanall = zeros(4,n,numMixturetemp,numStep,numMC);
ccovarall = zeros(4,n,n,numMixturetemp,numStep,numMC);
cweightall = zeros(4,numMixturetemp,numStep,numMC);

%% Monte Carlo simulation
% parfevalOnAll(@warning,0,'off','all'); % Temporary error off
for mc = 1:numMC 
    disptemp = ['Monete Carlo iteration: ', num2str(mc)];
    disp(disptemp)

    % Generate true states
    trueState = zeros(n,numStep);
    procNoise = sqrt(Q)*randn(n,numStep);
    trueState(:,1) = trueinitial+sqrt(P0)*randn(n,1);

    for i = 2:numStep
        trueState(:,i) = funsys(trueState(:,i-1),procNoise(i-1));
    end

    truePos = funpos(trueState);
    
    % Generate measurements
    measNoise = sqrt(R)*randn(nm,numStep);
    meas = funmeas(trueState,measNoise);

    % For evaluation
    error = zeros(9,n,numStep);
    chisq = zeros(9,numStep); % chi-squared statistic storage
    elpst = zeros(9,1);

    crcnt = zeros(9,1);
    divcnt = zeros(1,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);
    ccovarmc = zeros(4,n,n,numMixturetemp,numStep);
    cweightmc = zeros(4,numMixturetemp,numStep);

    %% For plot
    if figure_view
        fig2 = figure; 
        hold on; legend;
        plot(t,trueState(1,:),'--ok','DisplayName', 'True')
        xlabel('Time (sec)')
        ylabel('Estimated angle (rad)')

        fig3 = figure; 
        hold on; legend;
        plot(t,trueState(2,:),'--ok','DisplayName', 'True')
        xlabel('Time (sec)')
        ylabel('Estimated angular velocity (rad/s)')

        fig4 = figure; 
        hold on; legend;
        xlabel('Time (sec)')
        ylabel('|Estimation error| (rad)')
    end

    %% PGM-DS
    if comparison_PGM_DS
        %% Filter parameters
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point n + 1
        epsilon = 5;            % distance to determine if a point is a core point 3
        
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
    
        error(1,:,1) = abs(estState(:,1)-trueState(:,1));
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
        
        particle_pos = funpos(particles(:,:,1));
    
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
    %         frame(1) = getframe(gcf);
        end
    
        %% Main loop
        tStart1 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1));
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
    
            %%  Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end
    
            [particles,estState,temp_particle_var,...
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);
    
            %% For evaluation
            error(1,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(1,i) = error(1,:,i)*(temp_particle_var(:,:,i)\error(1,:,i)');
    
            for k = length(cweight)
                cmeanmc(1,:,k,i) = cmean(:,k);
                ccovarmc(1,:,:,k,i) = ccovar(:,:,k);
                cweightmc(1,k,i) = cweight(k,1);
            end
        
            particle_pos = funpos(particles(:,:,i));
    
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(1) = toc(tStart1);
        avgerr = mean(error(1,1,:));
    
        if avgerr < 0.5
            crcnt(1) = 1;
        end
    
        temperror = zeros(1,length(error(1,1,:)));
        temperror(:) = error(1,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'r','DisplayName', 'PGM-DS')
            figure(fig3); 
            plot(t,estState(2,:),'r','DisplayName', 'PGM-DS')
            figure(fig4);
            plot(t,temperror,'r','DisplayName', 'PGM-DS')
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM-DS','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% PGM-DU
    if comparison_PGM_DU
        %% Filter parameters
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point n + 1
        epsilon = 5;            % distance to determine if a point is a core point 3
        
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
        % particles(:,:,1) = [2*pi*rand(1,numParticles);initial+P0*randn(1,numParticles)];
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
    
        error(2,:,1) = abs(estState(:,1)-trueState(:,1));
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
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart2 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1));
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
            [particles,estState,temp_particle_var,...
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);
    
            %% For evaluation
            error(2,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(2,i) = error(2,:,i)*(temp_particle_var(:,:,i)\error(2,:,i)');
    
            for k = length(cweight)
                cmeanmc(2,:,k,i) = cmean(:,k);
                ccovarmc(2,:,:,k,i) = ccovar(:,:,k);
                cweightmc(2,k,i) = cweight(k,1);
            end  
    
            particle_pos = funpos(particles(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(2) = toc(tStart2);
        avgerr = mean(error(2,1,:));
        
        if avgerr < 0.5
            crcnt(2) = 1;
        end
        
        temperror = zeros(1,length(error(2,1,:)));
        temperror(:) = error(2,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'g','DisplayName', 'PGM-DU')
            figure(fig3);
            plot(t,estState(2,:),'g','DisplayName', 'PGM-DU')
            figure(fig4);
            plot(t,temperror,'g','DisplayName', 'PGM-DU')
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM-DU','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end
  
    %% PGM1
    if comparison_PGM1
        %% Filter parameters
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
    
        error(3,:,1) = abs(estState(:,1)-trueState(:,1));
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
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart3 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1));
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
    
            %%  Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end
    
            [particles,estState,temp_particle_var,...
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);
            
            %% For evaluation
            error(3,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(3,i) = error(3,:,i)*(temp_particle_var(:,:,i)\error(3,:,i)');
    
            for k = length(cweight)
                cmeanmc(3,:,k,i) = cmean(:,k);
                ccovarmc(3,:,:,k,i) = ccovar(:,:,k);
                cweightmc(3,k,i) = cweight(k,1);
            end
        
            particle_pos = funpos(particles(:,:,i));
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
    
        end
    
        elpst(3) = toc(tStart3);
        avgerr = mean(error(3,1,:));
    
        if avgerr < 0.5
            crcnt(3) = 1;
        end
    
        temperror = zeros(1,length(error(3,1,:)));
        temperror(:) = error(3,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'Color','#0072BD','DisplayName', 'PGM1')
            figure(fig3);
            plot(t,estState(2,:),'Color','#0072BD','DisplayName', 'PGM1')
            figure(fig4);
            plot(t,temperror,'Color','#0072BD','DisplayName', 'PGM1')
        end
        
        if recordVid_view
            writeObj = VideoWriter('PGM','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% PGM-UT
    if comparison_PGM1_UT
        %% Filter parameters
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
    
        error(4,:,1) = abs(estState(:,1)-trueState(:,1));
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
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart4 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1));
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
            [particles,estState,temp_particle_var,...
             cmean,ccovar,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);
    
            %% For evaluation
            error(4,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(4,i) = error(4,:,i)*(temp_particle_var(:,:,i)\error(4,:,i)');
    
            for k = length(cweight)
                cmeanmc(4,:,k,i) = cmean(:,k);
                ccovarmc(4,:,:,k,i) = ccovar(:,:,k);
                cweightmc(4,k,i) = cweight(k,1);
            end  
    
            particle_pos = funpos(particles(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(4) = toc(tStart4);
        avgerr = mean(error(4,1,:));
        
        if avgerr < 0.5
            crcnt(4) = 1;
        end
        
        temperror = zeros(1,length(error(4,1,:)));
        temperror(:) = error(4,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'Color','#EDB120','DisplayName', 'PGM1-UT')
            figure(fig3);
            plot(t,estState(2,:),'Color','#EDB120','DisplayName', 'PGM1-UT')
            figure(fig4);
            plot(t,temperror,'Color','#EDB120','DisplayName', 'PGM1-UT')
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM-UT','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% AKKF
    if comparison_AKKF
        %% Filter parameters
        AKKF.kernel = 1;        % "Quadratic" "Quartic" "Gaussian"
        AKKF.N_P = numParticles;
    
        AKKF.c = 1;
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
    
        particle_pos = funpos(AKKF.Z_P(:,:,1));
        
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart5 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                 AKKF.X_P(:,p,i) = funsys(AKKF.X_P_proposal(:,p,i-1),sqrt(Q)*randn(1));
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
            chisq(5,i) = error(4,:,i)*(temp_particle_var(:,:,i)\error(4,:,i)');
    
            particle_pos = funpos(AKKF.X_P(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(5) = toc(tStart5);
        avgerr = mean(error(5,1,:));
    
        if avgerr < 0.5
            crcnt(5) = 1;
        end
            
        temperror = zeros(1,length(error(5,1,:)));
        temperror(:) = error(5,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'m','DisplayName', 'AKKF')
            figure(fig3);
            plot(t,estState(2,:),'m','DisplayName', 'AKKF')
            figure(fig4);
            plot(t,temperror,'m','DisplayName', 'AKKF')
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
        ensembles(:,:,1) = initial+sqrt(P0)*randn(n,numEnsembles); 
        ensemble_var(:,:,1) = P0;
        
        estState(:,1) = mean(ensembles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(6,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(6,1) = error(6,:,1)*(temp_particle_var(:,:,1)\error(6,:,1)');
    
        particle_pos = funpos(ensembles(:,:,1));
        
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
        
        %% Main loop
        tStart6 = tic; 

        [estState,ensemble_var] = EnKF(numStep,n,nm,meas,numEnsembles,...
                                       estState,ensembles,ensemble_var,ensemble_mean,...
                                       Q,R,[],[],jacobianH,funsys,funmeas,0,0);
    
        %% For evaluation
        for i = 2:numStep
            temp_particle_var(:,:,i) = ensemble_var(:,:,i);
    
            error(6,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(6,i) = error(6,:,i)*(temp_particle_var(:,:,i)\error(6,:,i)');
    
            particle_pos = funpos(ensembles(:,:,i));
           
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(6) = toc(tStart6);
        avgerr = mean(error(6,1,:));
    
        if avgerr < 0.5
            crcnt(6) = 1;
        end
    
        temperror = zeros(1,length(error(6,1,:)));
        temperror(:) = error(6,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'c','DisplayName', 'EnKF')
            figure(fig3);
            plot(t,estState(2,:),'c','DisplayName', 'EnKF')
            figure(fig4);
            plot(t,temperror,'c','DisplayName', 'EnKF')
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
    
        error(7,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(7,1) = error(7,:,1)*(temp_particle_var(:,:,1)\error(7,:,1)');
    
        %% Main loop
        tStart7 = tic; 

        [estState,x,P] = EKF(numStep,n,nm,meas,estState,x,P,...
                             Q,R,F,[],[],jacobianH,G,funsys,funmeas,1,0,0);

        %% For evaluation
        for i = 2:numStep
            error(7,:,i) = x-trueState(:,i);
            chisq(7,i) = error(7,:,i)*(P\error(7,:,i)');   
        end
    
        elpst(7) = toc(tStart7);
        avgerr = mean(error(7,1,:));
    
        if avgerr < 0.5
            crcnt(7) = 1;
        end
    
        temperror = zeros(1,length(error(7,1,:)));
        temperror(:) = error(7,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'Color','#7E2F8E','DisplayName', 'EKF')
            figure(fig3);
            plot(t,estState(2,:),'Color','#7E2F8E','DisplayName', 'EKF')
            figure(fig4);
            plot(t,temperror,'Color','#7E2F8E','DisplayName', 'EKF')
        end
    end

    %% UKF
    if comparison_UKF
        %% Initialisation
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        x = initial;
        P = P0;
        
        estState(:,1) = x;
        temp_particle_var(:,:,1) = P;
    
        error(8,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(8,1) = error(8,:,1)*(temp_particle_var(:,:,1)\error(8,:,1)');
    
        %% Main loop
        tStart8 = tic; 

        [estState,x,P] = UKF(numStep,n,nm,meas,estState,x,P,...
                             numSigma,alpha,beta,lambda,...
                             Q,R,G,funsys,funmeas,0);
    
        %% For evaluation
        for i = 2:numStep
            error(8,:,i) = abs(x-trueState(:,i));
            chisq(8,i) = error(8,:,i)*(P\error(8,:,i)');    
    
            if any(eig(P) > 1.0e+04)
                divcnt = 1;
                break
            end
        end
        
        elpst(8) = toc(tStart8);
        avgerr = mean(error(8,1,:));
    
        if avgerr < 0.5
            crcnt(8) = 1;
        end
    
        temperror = zeros(1,length(error(8,1,:)));
        temperror(:) = error(8,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'Color','#4DBEEE','DisplayName', 'UKF')
            figure(fig3);
            plot(t,estState(2,:),'Color','#4DBEEE','DisplayName', 'UKF')
            figure(fig4);
            plot(t,temperror,'Color','#4DBEEE','DisplayName', 'UKF')
        end
    end

    %% SIR
    if comparison_SIR
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        weights = zeros(numParticles,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);    
        weights(:,1) = 1/numParticles*ones(numParticles,1);
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(9,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(9,1) = error(8,:,1)*(temp_particle_var(:,:,1)\error(8,:,1)');
    
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
        
        %% Main loop
        tStart9 = tic; 
    
        [estState,particles,weights] = ...
                SIR(numStep,n,nm,meas,numParticles,estState,particles,weights,...
                    Q,R,[],[],funsys,funmeas,0,[],0);
    
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
    
            error(9,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(9,i) = error(9,:,i)*(temp_particle_var(:,:,i)\error(9,:,i)');
    
            particle_pos = funpos(particles(:,:,i));
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(9) = toc(tStart9);
        avgerr = mean(error(1,:));
        
        if avgerr < 0.5
            crcnt(9) = 1;
        end
    
        temperror = zeros(1,length(error(9,1,:)));
        temperror(:) = error(9,1,:);
    
        if figure_view
            figure(fig2); 
            plot(t,estState(1,:),'Color','#77AC30','DisplayName', 'SIR')
            figure(fig3);
            plot(t,estState(2,:),'Color','#77AC30','DisplayName', 'SIR')
            figure(fig4);
            plot(t,temperror,'Color','#77AC30','DisplayName', 'SIR')
        end
    
        if recordVid_view
            writeObj = VideoWriter('SIR','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% End of parfor
    errorall(:,:,:,mc) = error;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    correctcnt(:,mc) = crcnt;
    divergecnt(:,mc) = divcnt;

    cmeanall(:,:,:,:,mc) = cmeanmc;
    ccovarall(:,:,:,:,:,mc) = ccovarmc;
    cweightall(:,:,:,mc) = cweightmc;
end

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
    armse(fil) = sum(rmse(fil,:))/numStep;  % PGM-UT/PGM/UKF/SIR
    cnttotal(fil) = sum(correctcnt(fil,:));
end

divall = sum(divergecnt(:,:));

% Plot
figure; hold on; legend;
plot(t,rmse(1,:),'r','DisplayName', 'PGM-DS')
plot(t,rmse(2,:),'g','DisplayName', 'PGM-DU')
plot(t,rmse(3,:),'Color','#0072BD','DisplayName', 'PGM1')
plot(t,rmse(4,:),'Color','#EDB120','DisplayName', 'PGM1-UT')
plot(t,rmse(5,:),'m','DisplayName', 'AKKF')
% plot(t,rmse(6,:),'c','DisplayName', 'EnKF')
% plot(t,rmse(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,rmse(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,rmse(9,:),'Color','#77AC30','DisplayName', 'SIR')

xlim([0 tf])
% ylim([-25 25])
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
plot(t,chinorm(1,:),'r','DisplayName', 'PGM-DS')
plot(t,chinorm(2,:),'g','DisplayName', 'PGM-DU')
plot(t,chinorm(3,:),'Color','#0072BD','DisplayName', 'PGM1')
plot(t,chinorm(4,:),'Color','#EDB120','DisplayName', 'PGM1-UT')
plot(t,chinorm(5,:),'m','DisplayName', 'AKKF')
% plot(t,chinorm(6,:),'c','DisplayName', 'EnKF')
% plot(t,chinorm(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,chinorm(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,chinorm(9,:),'Color','#77AC30','DisplayName', 'SIR')

xlim([0 tf])
ylim([0 15000])
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
plot(t,chifrac(1,:),'r','DisplayName', 'PGM-DS')
plot(t,chifrac(2,:),'g','DisplayName', 'PGM-DU')
plot(t,chifrac(3,:),'Color','#0072BD','DisplayName', 'PGM1')
plot(t,chifrac(4,:),'Color','#EDB120','DisplayName', 'PGM1-UT')
plot(t,chifrac(5,:),'m','DisplayName', 'AKKF')
% plot(t,chifrac(6,:),'c','DisplayName', 'EnKF')
% plot(t,chifrac(7,:),'Color','#7E2F8E','DisplayName', 'EKF')
plot(t,chifrac(8,:),'Color','#4DBEEE','DisplayName', 'UKF')
plot(t,chifrac(9,:),'Color','#77AC30','DisplayName', 'SIR')

xlim([0 tf])
% ylim([-25 25])
xlabel('Time (sec)')
ylabel('Fraction of Cases')
hold off;

elpadtavg = mean(elpsdtime,2);

%% Save variables
save('2_Pendulum_MC')