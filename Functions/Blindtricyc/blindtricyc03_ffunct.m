function [fk,Phik,Gammak] = blindtricyc03_ffunct(xk,wk,k,iflagfonly,...
                                                   ukhist,tkhist,bw)
%
%  Copyright (c) 2011 Mark L. Psiaki.  All rights reserved.  
%
%  "And whatever you do, whether in word or deed, do it all in the name of 
%   the Lord Jesus, giving thanks to God the Father through Him." 
%   (Colossians 3:17).
%
%
%  This function models the dynamics of the blind tricyclist
%  problem with M merry-go-round-riding friends who call to him
%  to give him relative bearing measurements periodically.
%
%      xkp1 = f(xk,wk,k;ukhist,tkhist,bw)
%
%  Inputs:
%
%    xk          The (3+2*M)-by-1 state vector of this system at sample  
%                time tk = tkhist(k+1,1), i.e., at sample number k.
%                xk(1,1) is the east position in meters, xk(2,1) is the
%                north position in meters, xk(3,1) is the yaw/heading
%                of the tricycle center line in radians and measured
%                from east, positive counter-clockwise so that a
%                heading of xk(3,1) = pi/2 radians is north, xk(3+mm,1)
%                for m = 1:M is the angle of the mth merry-go-round-
%                riding friend's merry-go-round measured counter-clockwise
%                from east in radians, and xk(3+M+m,1) for m = 1:M is 
%                the rotation rate of the mth merry-go-round riding 
%                friend's merry-go-round in radians/sec with
%                counter-clockwise motion corresponding to positive
%                rates.
%
%    wk          The 5-by-1 process noise vector of this system that
%                operates from time tk = tkhist(k+1,1) to time tkp1 = ...
%                tkhist(k+2,1).  wk(1,1) is the incremental speed noise
%                above the value in ukhist(k+1,1) in meters/sec,
%                wk(2,1) is the incremental front-wheel steer angle
%                noise above the value in ukhist(k+1,2) in radians, 
%                wk(3,1) is the eastward slip-rate noise in m/sec,
%                wk(4,1) is the northward slip-rate noise in m/sec, and
%                wk(5,1) is the yaw/heading slip-rate noise in rad/sec,
%                positive clockwise.
%
%    k           The index of the current sample time.  It selects the
%                dynamic propagation interval as being the one from
%                time tk = tkhist(k+1,1) to time tkp1 = tkhist(k+2,1).
%
%    iflagfonly  A flag that tells whether to output only fk (iflagfonly
%                = 1) or to also output the Jacobian matrices
%                Phik and Gammak (iflagfonly = 0).  The iflagfonly == 1
%                option may be convenient for use when only
%                performing a simulation or when doing sigma-points
%                filtering or particle filtering because it
%                saves execution time by not computing the
%                unneeded Phik and Gammak matrices.
%
%    ukhist      The N-by-2 array of speed/steer-angle commands for 
%                the time intervals defined in tkhist.  Vk = ukhist(k+1,1)
%                is the commanded forward speed of the mid-point between 
%                the rear wheels, in m/sec, for the time interval from 
%                tk = tkhist(k+1,1) to time tkp1 = tkhist(k+2,1).
%                The corresponding front-wheel steer angle in radians
%                during this time interval is gammak = ukhist(k+1,2).
%                It is defined as positive for leftward steer, i.e., for
%                an angle that would cause a positive, i.e., counter-
%                clockwise, yaw rate.  ukhist(k+1,2) = 0 corresponds
%                to a command for straight-line motion.
%
%    tkhist      The (N+1)-by-1 vector of sample times in seconds.  tk =
%                tkhist(k+1,1) is the time of the kth sample.  These 
%                are times at which bearing measurements may or 
%                may not be taken.  
%
%                Note that this model assumes constant velocity
%                and steering commands for each interval between
%                sample times in tkhist and constant process noise
%                elements in wk over such intervals.
%
%    bw          The wheel base of the tricycle, in meters.  This is the
%                distance between the ground-contact point of the front
%                wheel and the mid-point between the ground contact
%                points of the rear wheels.
%
%  Outputs:
%
%    fk          The (3+2*M)-by-1 dynamics function that gives xkp1 in the
%                difference equation model.  Its elements have the
%                same units as the corresponding elements of xk.
%
%    Phik        The (3+2*M)-by-(3+2*M) partial derivative of fk with
%                respect to xk.  This is an empty array on output if 
%                iflagfonly = 1.
%
%    Gammak      The (3+2*M)-by-5 partial derivative of fk with
%                respect to wk.  This is an empty array on output
%                if iflagfonly = 1.
%

%
%  Determine the state dimension and the number of friends on
%  merry-go-rounds.
%
   nx = size(xk,1);
   rM = (nx - 3)/2;
   M = round(rM);
   if abs(M - rM) > 0.1
      error(['Error in ffunct_blindtricyc03.m: The dimension of',...
             ' the state, nx, is not odd.  Therefore, the number of',...
             ' merry-go-round-ridinig friends is not an integer.'])
   end
%
%  Compute the time interval.
%
   kp1 = k + 1;
   kp2 = kp1 + 1;
   tk = tkhist(kp1,1);
   tkp1 = tkhist(kp2,1);
   deltk = tkp1 - tk;
   if deltk <= 0
      error(['Error in ffunct_blindtricyc03.m: Found non-positive',...
             ' dynamic propagation interval ',int2str(k),'.'])
   end
%
%  Retrieve the command and add to it the first two process noise
%  vector components.
%
   ukpw12k = ukhist(kp1,:)' + wk(1:2,1);
   Vtotk = ukpw12k(1,1);
   gammatotk = ukpw12k(2,1);
%
%  Compute the sinc and cinc (= (cos(aeta)-1)/aeta) functions.
%
   deltkobw = deltk/bw;
   tangammatotk = tan(gammatotk);
   aetak = deltkobw*Vtotk*tangammatotk;
   if iflagfonly == 1
      iflagsinckcinckdervs = 0;
   else
      iflagsinckcinckdervs = 1;
   end
   [sinck,cinck,dsinckdaetak,dcinckdaetak] = ...
                     blindtricyc03_sacafunct(aetak,iflagsinckcinckdervs);
%
%  Compute the sine and cosine of the heading angle.
%
   thetak = xk(3,1);
   sinthetak = sin(thetak);
   costhetak = cos(thetak);
%
%  Compute the first three states at time tkp1, ignoring the
%  slip from wk(3:5,1).
%
   dellengthk = Vtotk*deltk;
   dXkp1dirdum = sinthetak*cinck + costhetak*sinck;
   deltaXk = dellengthk*dXkp1dirdum;
   Xkp1 = xk(1,1) + deltaXk;
   dYkp1dirdum = -costhetak*cinck + sinthetak*sinck;
   deltaYk = dellengthk*dYkp1dirdum;
   Ykp1 = xk(2,1) + deltaYk;
   thetakp1 = thetak + aetak;
%
%  Compute fk.
%
   fk = zeros(nx,1);
   fk(1:3,1) = [Xkp1;Ykp1;thetakp1] + wk(3:5,1)*deltk;
   indexvecmerrygoroundangles = 3 + [1:M]';
   indexvecmerrygoroundrates = indexvecmerrygoroundangles + M;
   alphadotveck = xk(indexvecmerrygoroundrates,1);
   fk(indexvecmerrygoroundangles,1) = xk(indexvecmerrygoroundangles,1) + ...
                alphadotveck*deltk;
   fk(indexvecmerrygoroundrates,1) = alphadotveck;
%
%  Return if only fk is needed.
%
   if iflagfonly == 1
      Phik = [];
      Gammak = [];
      return
   end
%
%  Compute the required Jacobian matrices.
%
   Phik = zeros(nx,nx);
   Phik(1:2,1:2) = eye(2);
   Phik(1:3,3) = [(-deltaYk);deltaXk;1];
   Phik(indexvecmerrygoroundangles,indexvecmerrygoroundangles) = eye(M);
   Phik(indexvecmerrygoroundangles,indexvecmerrygoroundrates) = ...
                     eye(M)*deltk;
   Phik(indexvecmerrygoroundrates,indexvecmerrygoroundrates) = eye(M);
%
   Gammak = zeros(nx,5);
   dVtotk_dwk1 = 1;
   dgammatotk_dwk2 = 1;
%
   dtangammatotk_dgammatotk = 1 + tangammatotk^2;
   dtangammatotk_dwk2 = dtangammatotk_dgammatotk*dgammatotk_dwk2;
   daetak_dwk12 = deltkobw*[(dVtotk_dwk1*tangammatotk),...
                                (Vtotk*dtangammatotk_dwk2)];
   dsinck_dwk12 = dsinckdaetak*daetak_dwk12;
   dcinck_dwk12 = dcinckdaetak*daetak_dwk12;
%
   dXkp1_dwk12 = dellengthk*(sinthetak*dcinck_dwk12 + ...
                             costhetak*dsinck_dwk12);
   dYkp1_dwk12 = dellengthk*(-costhetak*dcinck_dwk12 + ...
                              sinthetak*dsinck_dwk12);
   dthetakp1_dwk12 = daetak_dwk12;
%
   Gammak(1:3,1:2) = [dXkp1_dwk12;dYkp1_dwk12;dthetakp1_dwk12];
   Gammak(1:2,1) = Gammak(1:2,1) + deltk*[dXkp1dirdum;dYkp1dirdum];
   Gammak(1:3,3:5) = eye(3)*deltk;