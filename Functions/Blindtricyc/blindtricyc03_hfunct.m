function [hk,Hk,mykvec] = blindtricyc03_hfunct(xk,k,iflaghonly,br,...
                                               Xvec,Yvec,rhovec,...
                                               imykflaghist,yaltkhist)
%
%  Copyright (c) 2011 Mark L. Psiaki.  All rights reserved.  
%
%  "And whatever you do, whether in word or deed, do it all in the name of 
%   the Lord Jesus, giving thanks to God the Father through Him." 
%   (Colossians 3:17).
%
%
%  This function models the measurements of the blind tricyclist problem
%  with M merry-go-round-riding friends who call to him
%  to give him relative bearing measurements intermittently.
%
%  Inputs:
%
%    xk            The (3+2*M)-by-1 state vector of this system at sample  
%                  time tk = tkhist(k+1,1), i.e., at sample number k.
%                  xk(1,1) is the east position in meters, xk(2,1) is the
%                  north position in meters, xk(3,1) is the yaw/heading
%                  of the tricycle center line in radians and measured
%                  from east, positive counter-clockwise so that a
%                  heading of xk(3,1) = pi/2 radians is north, xk(3+m,1)
%                  for m = 1:M is the angle of the mth merry-go-round-
%                  riding friend's merry-go-round measured counter-
%                  clockwise from east in radians, and xk(3+M+m,1) 
%                  for m = 1:M is the rotation rate of the mth merry-
%                  go-round riding friend's merry-go-round in 
%                  radians/sec with counter-clockwise motion 
%                  corresponding to positive rates.
%
%    k             The index of the current sample time.
%
%    iflaghonly    A flag that tells whether to output only hk 
%                  (iflaghonly = 1) or to also output the Jacobian 
%                  matrix Hk (iflaghonly = 0).  The iflaghonly == 1
%                  option may be convenient for use when only
%                  performing a simulation or when doing sigma-points
%                  filtering or particle filtering because it
%                  saves execution time by not computing the
%                  unneeded Hk matrix.
%
%    br            The distance of the rider forward of the mid-
%                  point between the rear-wheel ground contact points,
%                  in meters.  This defines the position from which
%                  the relative bearings to the merry-go-round-riding
%                  friends are measured.
%
%    Xvec          The M-by-1 vector of east positions of the 
%                  merry-go-round centers, in meters.
%
%    Yvec          The M-by-1 vector of north positions of the 
%                  merry-go-round centers, in meters.
%
%    rhovec        The M-by-1 vector of merry-go-round radii, in meters.
%
%    imykflaghist  The (N+1)-by-M array of measurement flags.  
%                  imykflagvec = imykflaghist(k+1,:)' is the M-by-1
%                  vector of measurement flags for sample k.  If
%                  imykflagvec(m,1) == 1, then the mth merry-go-round
%                  rider gives a shout and has his relative bearing
%                  measured on sample k.  Otherwise, there is no
%                  measurement from that merry-go-round rider on
%                  sample k.  The total number of relative bearing 
%                  measurements on sample k is lk = sum(imykflagvec == 1).
%
%    yaltkhist     The optional input that, if entered as an (N+1)-by-M 
%                  array, is presumed to give actual measurements of
%                  the hk outputs for the entries whose corresponding
%                  elements of imykflaghist are equal to 1.  
%                  If mykvec = find(imykflagvec == 1) and if
%                  yk = yaltkhist(k+1,mykvec)', then
%                  yk is used to determine multiples 
%                  of 2*pi to add to the raw relative bearing 
%                  measurements so that hk is within +/- pi
%                  of yk.  This effectively transforms hk
%                  into hfixk.
%
%  Outputs:
%
%    hk            The lk-by-1 measurement function that models yk.
%                  hk(jj,1) is the relative bearing angle to the 
%                  mykvec(jj,1) merry-go-round riding 
%                  friend, the one whose current merry-go-round angle 
%                  is xk(3+mykvec(jj,1),1),  where
%                  mykvec = find(imykflagvec == 1).  The units
%                  of hk are radians.  The bearings are measured
%                  relative to the straight-ahead direction of the
%                  tricycle.  If the optional input yaltkhist is
%                  input, then hk actually outputs hfixk, the
%                  form of hk that has had multiples of 2*pi
%                  added to or subtracted from each element so 
%                  that each one differs from yk = yaltkhist(k+1,mykvec)'
%                  by a magnitude of no more than pi.
%
%    Hk            The lk-by-(3+2*M) partial derivative of hk with
%                  respect to xk.  This is an empty array on output 
%                  if iflaghonly = 1.
%
%    mykvec        The lk-by-1 vector of measurement indices
%                  that indicate the merry-go-rounds and merry-go-
%                  round-riding friends associated with the 
%                  measurements in hk.
%

%
%  Retrieve the vector of measurement flags and determine the
%  number of measurements.  Also, determine the number of merry-go-
%  rounds.
%
   kp1 = k + 1;
   imykflagvec = imykflaghist(kp1,:)';
   M = size(imykflagvec,1);
   mykvec = find(imykflagvec == 1);
   lk = size(mykvec,1);
%
%  Return if no measurements are to be made.
%
   if lk == 0
      hk = [];
      Hk = [];
      mykvec = [];
      return
   end
%
%  Check M and the state vector dimension.
%
   nx = size(xk,1);
   if nx ~= (3 + 2*M)
      error(['Error in hfunct_blindtricyc03.m: The dimension of',...
             ' the state, nx, does not equal 3+2*M, where M is',...
             ' the number of merry-go-rounds & merry-go-round',...
             ' riders.'])
   end
%
%  Detrmine the location of the rider and of the merry-go-round
%  riders who are call to the rider at the current sample time.
%
   thetak = xk(3,1);
   costhetak = cos(thetak);
   sinthetak = sin(thetak);
   delXriderk = br*costhetak;
   Xriderk = xk(1,1) + delXriderk;
   delYriderk = br*sinthetak;
   Yriderk = xk(2,1) + delYriderk;
   mykvecp3 = mykvec + 3;
   phimerrgoroundveck = xk(mykvecp3,1);
   Xvec0merrygoroundk = Xvec(mykvec,1);
   Yvec0merrygoroundk = Yvec(mykvec,1);
   rhovecmerrygoroundk = rhovec(mykvec,1);
   cosphimerrgoroundveck = cos(phimerrgoroundveck);
   sinphimerrgoroundveck = sin(phimerrgoroundveck);
   delXvecmerrygoroundk = rhovecmerrygoroundk.*cosphimerrgoroundveck;
   Xvecmerrygoroundk = Xvec0merrygoroundk + delXvecmerrygoroundk;
   delYvecmerrygoroundk = rhovecmerrygoroundk.*sinphimerrgoroundveck;
   Yvecmerrygoroundk = Yvec0merrygoroundk + delYvecmerrygoroundk;
%
%  Compute the absolute bearings to the merry-go-round riders.
%
   deltaXvecmerrygoroundk = Xvecmerrygoroundk - Xriderk;
   deltaYvecmerrygoroundk = Yvecmerrygoroundk - Yriderk;
   psiabsvecmerrygoroundk = atan2(deltaYvecmerrygoroundk,...
                                  deltaXvecmerrygoroundk);
%
%  Compute the relative bearings to the merry-go-round riders
%  and normalize them to get the output vector.
%
   psivecmerrygoroundk = psiabsvecmerrygoroundk - thetak;
   twopi = 2*pi;
   ootwopi = 1/twopi;
   psivecmerrygoroundk = psivecmerrygoroundk - ...
                    twopi*round(ootwopi*psivecmerrygoroundk);
%
%  Perform extra rounding if called for.
%
   if nargin > 8
      [ndumz,mdumz] = size(yaltkhist);
      if mdumz == M
         ndumi = size(imykflaghist,1);
         if ndumz == ndumi
            yk = yaltkhist(kp1,mykvec)';
            psivecmerrygoroundk_pre = yk;
            psivecmerrygoroundk = psivecmerrygoroundk - ...
                    twopi*round(ootwopi*(psivecmerrygoroundk - ...
                                         psivecmerrygoroundk_pre));
         end
      end
   end
   hk = psivecmerrygoroundk;
%
%  Return if only hk is needed.
%
   if iflaghonly == 1
      Hk = [];
      return
   end
%
%  Compute the required Jacobian matrix.
%
   dXriderk_dx123k = [1,0,(-delYriderk)];
   dYriderk_dx123k = [0,1,delXriderk];
   ddeltaXvecmerrygoroundk_dxk = zeros(lk,nx);
   ddeltaYvecmerrygoroundk_dxk = zeros(lk,nx);
   ones_lk = ones(lk,1);
   ddeltaXvecmerrygoroundk_dxk(:,1:3) = ones_lk*(-dXriderk_dx123k);
   ddeltaYvecmerrygoroundk_dxk(:,1:3) = ones_lk*(-dYriderk_dx123k);
   ddeltaXvecmerrygoroundk_dxk(:,mykvecp3) = ...
                                  diag(-delYvecmerrygoroundk);
   ddeltaYvecmerrygoroundk_dxk(:,mykvecp3) = ...
                                  diag(delXvecmerrygoroundk);
   denvec = (deltaXvecmerrygoroundk.^2 + deltaYvecmerrygoroundk.^2).^(-1);
   dXvecfac = - deltaYvecmerrygoroundk.*denvec;
   dYvecfac = deltaXvecmerrygoroundk.*denvec;
   dpsivecmerrygoroundk_dxk = ...
                 diag(dXvecfac)*ddeltaXvecmerrygoroundk_dxk + ...
                 diag(dYvecfac)*ddeltaYvecmerrygoroundk_dxk;
   dpsivecmerrygoroundk_dxk(:,3) = dpsivecmerrygoroundk_dxk(:,3) - ...
                                   ones_lk;
   Hk = dpsivecmerrygoroundk_dxk;