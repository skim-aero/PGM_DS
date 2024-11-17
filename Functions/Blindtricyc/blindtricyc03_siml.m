function [xkhist_true,yaltkhist,yaltkhist_true,wkhist,x0hat,Ralt,...
                         mykveccellhist,Hobsmat] = ...
               blindtricyc03_siml(x0,ukhist,tkhist,bw,br,...
                                  Xvec,Yvec,rhovec,sigmavec,...
                                  imykflaghist,P0,Q,...
                                  iflagobsmat,iseedrandn)
%
%  Copyright (c) 2011 Mark L. Psiaki.  All rights reserved.  
%
%  "And whatever you do, whether in word or deed, do it all in the name of 
%   the Lord Jesus, giving thanks to God the Father through Him." 
%   (Colossians 3:17).
%
%
%  This function simulates the state time history and the measurement
%  time history of the blind tricyclist who senses tricycle-relative 
%  bearings to friends who are riding on merry-go-rounds with 
%  known locations and radii, but with unknown initial rotation
%  angles and unknown constant rotation rates.  This is for
%  a recursive Kalman filtering problem simulation that has
%  process noise.
%
%  This function uses ffunct_blindtricyc03.m in order to simulate
%  the tricyclist's and the merry-go-round's dynamics, and it uses
%  blindtricyc03_haltfunct.m in order to simulate the zero-padded
%  measurements, with zero-padding included as place holders for
%  any unavailable relative bearing measurements.
%
%  Inputs:
%
%    x0              The (3+2*M)-by-1 state vector of this system at  
%                    sample time t0 = tkhist(1,1), i.e., at sample
%                    number k = 0.  x0(1,1) is the east position in 
%                    meters, x0(2,1) is the north position in meters, 
%                    x0(3,1) is the yaw/heading of the tricycle center 
%                    line in radians and measured from east, positive 
%                    counter-clockwise so that a heading of  
%                    x0(3,1) = pi/2 radians is north, x0(3+m,1) for 
%                    m = 1:M is the angle of the mth merry-go-round-
%                    riding friend's merry-go-round measured counter-
%                    clockwise from east in radians, and x0(3+M+m,1)
%                    for m = 1:M is the rotation rate of the mth 
%                    merry-go-round riding friend's merry-go-round 
%                    in radians/sec with counter-clockwise motion 
%                    corresponding to positive rates.
%
%    ukhist          The N-by-2 array of speed/steer-angle commands
%                    for the time intervals defined in tkhist.  
%                    Vk = ukhist(k+1,1) is the commanded forward speed of  
%                    the mid-point between the rear wheels, in m/sec,
%                    for the time interval from tk = tkhist(k+1,1) to 
%                    time tkp1 = tkhist(k+2,1).  The corresponding
%                    front-wheel steer angle in radians during this 
%                    time interval is gammak = ukhist(k+1,2).  It is  
%                    defined as positive for leftward steer, i.e., for 
%                    an angle that would cause a positive, i.e., 
%                    counter-clockwise, yaw rate.  
%                    ukhist(k+1,2) = 0 corresponds to a command 
%                    for straight-line motion.
%
%    tkhist          The (N+1)-by-1 vector of sample times in seconds.  
%                    tk = tkhist(k+1,1) is the time of the kth sample.   
%                    These are times at which bearing measurements may 
%                    (or may not) be taken.  
%
%                    Note that this model assumes constant velocity
%                    and steering commands for each interval between
%                    sample times in tkhist.
%
%    bw              The wheel base of the tricycle, in meters.  This
%                    is the distance between the ground-contact point
%                    of the front wheel and the mid-point between the
%                    ground contact points of the rear wheels.
%
%    br              The distance of the rider forward of the mid-
%                    point between the rear-wheel ground contact points,
%                    in meters.  This defines the position from which
%                    the relative bearings to the merry-go-round-riding
%                    friends are measured.
%
%    Xvec            The M-by-1 vector of east positions of the 
%                    merry-go-round centers, in meters.
%
%    Yvec            The M-by-1 vector of north positions of the 
%                    merry-go-round centers, in meters.
%
%    rhovec          The M-by-1 vector of merry-go-round radii, in
%                    meters.
%
%    sigmavec        The M-by-1 vector of relative bearing angle 
%                    measurement error standard deviations in radians.
%
%    imykflaghist    The (N+1)-by-M array of measurement flags.  
%                    imykflagvec = imykflaghist(k+1,:)' is the M-by-1 
%                    vector of measurement flags for sample k.  If
%                    imykflagvec(m,1) == 1, then the mth merry-go-round
%                    rider gives a shout and has his relative bearing
%                    measured on sample k.  Otherwise, there is no
%                    measurement from that merry-go-round rider on
%                    sample k.  The total number of relative bearing 
%                    measurements on sample k is lk = ...
%                    sum(imykflagvec == 1).  Note that 
%                    imykflaghist(1,:) = zeros(1,M) is required 
%                    because of the assumption that there are 
%                    no measurements at sample k = 0.
%
%    P0              The (3+2*M)-by-(3+2*M) a posteriori state
%                    estimation error covariance matrix
%                    at time t0 = tkhist(1,1).  The units of 
%                    P0(ii,jj) are the units of x0(ii,1)
%                    multiplied by the units of x0(jj,1).
%
%    Q               The 5-by-5 a priori covariance matrix 
%                    for each process noise vector wk = wkhist(k+1,:)
%                    for k = 0, ..., (N-1).  The units of 
%                    Q(ii,jj) are the units of wk(ii,1)
%                    multiplied by the units of wk(jj,1).
%
%    iflagobsmat     An input flag that tells whether (iflagobsmat == 1)
%                    or not (iflagobsmat == 0) to compute and
%                    output the large observability matrix
%                    Hobsmat whose columns must be linearly
%                    independent in order for this system to
%                    be locally observable.
%
%    iseedrandn      The integer seed for the Gaussian random number
%                    generator that generates the measurement noise
%                    and process noise time histories and the
%                    initial estimation error in x0hat.
%
%  Outputs:
%
%    xkhist_true     The (N+1)-by-(3+2*M) array that stores the truth
%                    state time history at the times in tkhist. 
%                    xk = xkhist_true(k+1,:)' is the (3+2*M)-by-1 truth 
%                    state vector at time tk = tkhist(k+1,1).  The
%                    definitions of the individual elements of
%                    xk are the same as the definitions of the
%                    individual elements of x0, except that they
%                    apply at time tk rather than at time t0.
%
%    yaltkhist       The (N+1)-by-M array, of 
%                    actual measurements of the tricycle-
%                    relative bearings for the entries whose 
%                    corresponding elements of imykflaghist are equal
%                    to 1.   If imykflagvec = imykflaghist(k+1,:)' 
%                    and if mykvec = find(imykflagvec == 1), then
%                    yk = yaltkhist(k+1,mykvec)' is the
%                    vector of measured relative bearings, in radians,
%                    at time tk = tkhist(k+1,1) for the merry-go-
%                    round riders whose identification indices
%                    are contained in mykvec.
%
%    yaltkhist_true  The (N+1)-by-M array of truth values of the 
%                    measurements of the tricycle-relative bearings 
%                    in yaltkhist.  The only difference between 
%                    the elements of this array and those of 
%                    yaltkhist are that this array's measurements 
%                    do not include any measurement noise.
%
%    wkhist          The N-by-5 array that stores the truth process 
%                    noise time history.  wk = wkhist(k+1,:)' is the 
%                    5-by-1 process noise vector that acts from 
%                    time tk = tkhist(k+1,1) to tkp1 = tkhist(k+2,1).
%                    The column wkhist(:,1) gives the difference 
%                    between the true speed time history and the 
%                    commanded speed time history in ukhist(:,1) in
%                    units of meters/sec.  The column wkhist(:,2)  
%                    gives the difference between the true steer
%                    angle time history and the commanded steer 
%                    angle time history in ukhist(:,2) in units
%                    of radians.  The columns
%                    wkhist(:,3) and wkhist(:,4) give, respectively,
%                    the eastward slip rate and the northward slip
%                    rate of the mid-point between the real-wheel
%                    contact points in m/sec.  The column wkhist(:,5)
%                    gives the yaw/heading slip rate in rad/sec.
%
%    x0hat           The (3+2*M)-by-1 a posteriori estimate of
%                    x0 = xkhist_true(1,:)'.  This vector has the same
%                    definitions and units as x0.  It should be used
%                    in conjunction with P0 to initialize
%                    each Kalman filter, be it an EKF, a UKF,
%                    a BSEKF, or a particle filter.
%
%    Ralt            = diag(sigmavec.^2), the M-by-M measurement
%                    noise covariance matrix, in units of radians^2.
%
%    mykveccellhist  The (N+1)-by-1 cell array of index vectors that
%                    define the actual measurements.  mykvec = ...
%                    mykveccellhist{k+1,1} = find(imykflaghist(k+1,:)')
%                    is an lk-by-1 dimensional index vector
%                    that contains the indices, in the range 
%                    1:M, of the merry-go-round-riding friends 
%                    for which relative bearing measurements 
%                    are available at sample time 
%                    tk = tkhist(k+1,1).
%
%    Hobsmat         The (M*N)-by-(3+2*M) observability
%                    matrix for the linearized observability
%                    analysis.  This will be output only
%                    if iflagobsmat == 1 on input.  Otherwise,
%                    an empty array will be output.  The
%                    rank of this matrix must be (3+2*M)
%                    in order for this system to be locally
%                    observable.  This is effectively
%                    the discrete-time observability
%                    matrix for the local linearization
%                    of this system, and Hobsmat'*Hobsmat is
%                    the observability Gramian for the
%                    corresponding time-varying linearized
%                    system model.
%                    
%

%
%  Seed the Gaussian random number generator.
%
%    randn('state',iseedrandn);
%
%  Determine the number of states, the number of merry-go-rounds & 
%  friends, and the number of samples.
%
   N = size(ukhist,1);
   Np1 = size(tkhist,1);
   if Np1 ~= (N + 1)
      error(['Error in siml_blindtricyc03.m: Inconsistency between',...
             ' ukhist and tkhist as to the number of samples in the',...
             ' estimation problem.'])
   end
   M = size(Xvec,1);
   if (size(Yvec,1) ~= M) | (size(rhovec,1) ~= M) | ...
           (size(sigmavec,1) ~= M)
      error(['Error in siml_blindtricyc03.m: Inconsistency between',...
             ' number of merry-go-rounds/friends in Xvec, Yvec,',...
             ' rhovec, and sigmavec.'])
   end
   if (size(imykflaghist,1) ~= Np1) | (size(imykflaghist,2) ~= M)
      error(['Error in siml_blindtricyc03.m: Inconsistency in',...
             ' row or column dimension of imykflaghist.'])
   end
   nx = size(x0,1);
   if nx ~= (3 + 2*M)
      error(['Error in siml_blindtricyc03.m: The row dimension of',...
             ' the initial state vector x0 does not equal 3+2*M,',...
             ' where M is the number of merry-go-rounds &',...
             ' merry-go-round riders.'])
   end
%
%  Check that there are no measurements for the initial
%  sample interval.
%
   if max(abs(imykflaghist(1,:))) > 0
      error(['Error in siml_blindtricyc03.m: The first row of',...
             ' imykflaghist contains non-zero elements, which',...
             ' indicate measurements at sample k = 0, which',...
             ' are forbidden.'])
   end
%
%  Determine the total number of measurements.
%
   Nmeastot = sum(imykflaghist(:) == 1);
   if Nmeastot == 0
      disp('Warning in siml_blindtricyc03.m: Zero measurements')
      disp(' are simulated.')
   end
%
%  Set up the output arrays mostly with dummy zero values.  Set up
%  the actual process noise time history.
%
   xkhist_true = zeros(Np1,nx);
   yaltkhist = zeros(Np1,M);
   yaltkhist_true = zeros(Np1,M);
   mykveccellhist = cell(Np1,1);
   if norm(Q) == 0
      Rwwinv = 0*Q;
   else
      Rwwinv = chol(Q)';
   end
   wkhisttr = Rwwinv*randn(5,N);
   wkhist = wkhisttr';
   clear wkhisttr
%   
%  Compute the a posteriori initial state estimate.
%  
   if norm(P0) == 0
      Rxx0hatinv = 0*P0;
   else
      Rxx0hatinv = chol(P0)';
   end
   x0hat = x0 + Rxx0hatinv*randn(nx,1);
%
%  Prepare for the loop that performs the measurement simulation
%  and the dynamic propagation, one sample and sample interval
%  per iteration.  Also, set up the local observability matrix
%  output if that is called for.
%
   xk = x0;
   ibigmeasdone = 0;
   if iflagobsmat == 1
      iflaghonly = 0;
      iflagfonly = 0;
      Hobsmat = zeros((M*N),(3+2*M));
      Phikto0 = eye((3+2*M));
      iHobsmatrowvec = ((1:M)') - M;
   else
      iflaghonly = 1;
      iflagfonly = 1;
      Hobsmat = [];
   end
%
%  This is the main loop that performs the measurement simulation
%  and the dynamic propagation, one sample and sample interval
%  per iteration.
%
   for k = 0:N
%
%  Store the current state.
%
      kp1 = k + 1;
      xkhist_true(kp1,:) = xk';
%
%  Simulate the truth measurements and the noisy measurements.
%
      [haltk,Haltk,mykvec] = ...
                      blindtricyc03_haltfunct(xk,k,iflaghonly,br,...
                                              Xvec,Yvec,rhovec,...
                                              imykflaghist);
      mykveccellhist{kp1,1} = mykvec;
      nualtk = zeros(M,1);
      lk = size(mykvec,1);
      ibigmeasdone = ibigmeasdone + lk;
      if lk > 0
         nualtk(mykvec,1) = randn(lk,1).*sigmavec(mykvec,1);
      end
      yaltkhist(kp1,:) = (haltk + nualtk)';
      yaltkhist_true(kp1,:) = haltk';
%
%  Populate the rows of the discrete-time observability matrix
%  if that calculation is called for
%
      if iflagobsmat == 1
         if k > 0
            Hobsmat(iHobsmatrowvec,:) = Haltk*Phikto0;
         end
         iHobsmatrowvec = iHobsmatrowvec + M;
      end
%
%  Dynamically propagate the state to the next sample interval,
%  and re-name it as xk in preparation for the next iteration of
%  the main loop.
%
      if k < N
         wk = wkhist(kp1,:)';
         [fk,Phik] = blindtricyc03_ffunct(xk,wk,k,iflagfonly,...
                                          ukhist,tkhist,bw);
         xk = fk;
         if iflagobsmat == 1
            Phikto0 = Phik*Phikto0;
         end
      end
   end
%
%  Check measurement count consistency.
%
   if ibigmeasdone ~= Nmeastot
      disp('Warning in siml_blindtricyc03.m: Inconsistency in the')
      disp(' total count of measurements.')
   end
%
%  Assign the measurement error covariance matrix output.
%
   Ralt = diag(sigmavec.^2);