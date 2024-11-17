function [haltk,Haltk,mykvec] = ...
                       blindtricyc03_haltfunct(xk,k,iflaghonly,br,...
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
%  to give him relative bearing measurements periodically.
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
%                  for m = 1:M is the rotation rate of the mth 
%                  merry-go-round riding friend's merry-go-round
%                  in radians/sec with counter-clockwise motion 
%                  corresponding to positive rates.
%
%    k             The index of the current sample time.
%
%    iflaghonly    A flag that tells whether to output only haltk 
%                  (iflaghonly = 1) or to also output the Jacobian 
%                  matrix Haltk (iflaghonly = 0).  The iflaghonly == 1
%                  option may be convenient for use when only
%                  performing a simulation or when doing sigma-points
%                  filtering or particle filtering because it
%                  saves execution time by not computing the
%                  unneeded Haltk matrix.
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
%                  sample k.  The total number of non-trivial 
%                  relative bearing measurements on sample k is
%                  lk = sum(imykflagvec == 1).
%
%    yaltkhist     The optional input that, if entered as an (N+1)-by-M 
%                  array, is presumed to give actual measurements of
%                  the haltk outputs for the entries whose corresponding
%                  elements of imykflaghist are equal to 1.  
%                  If mykvec = find(imykflagvec == 1) and if
%                  yk = yaltkhist(k+1,mykvec)', then yk is used
%                  to determine multiples of 2*pi to add to the
%                  raw relative bearing measurements so that
%                  haltk(mykvec,1) is within +/- pi of yk.
%
%  Outputs:
%
%    haltk         The M-by-1 measurement function that gives the
%                  measurement model of how yaltk depends on the state.
%                  haltk(m,1) is the relative bearing angle to the 
%                  mth merry-go-round riding friend, the one whose 
%                  current merry-go-round angle is xk(3+m,1),
%                  if imykflagvec(m,1) == 1, otherwise it is 
%                  zero-valued.  The units of haltk  
%                  are radians.  The bearings are measured
%                  relative to the straight-ahead direction of the
%                  tricycle.  If the optional input yaltkhist is
%                  input, then haltk actually outputs haltfixk, the
%                  form of haltk that has had multiples of 2*pi
%                  added to or subtracted from each element so 
%                  that each element of haltk(mykvec,1) differs 
%                  from yk = yaltkhist(k+1,mykvec)' by a magnitude 
%                  of no more than pi.
%
%    Haltk         The M-by-(3+2*M) partial derivative of haltk with
%                  respect to xk.  This is an empty array on output 
%                  if iflaghonly = 1.
%
%    mykvec        The lk-by-1 vector of measurement indices
%                  that indicate the merry-go-rounds and merry-go-
%                  round-riding friends associated with the 
%                  non-trivial measurements in haltk.
%

%
%  This function has been created by calling hfunct_blindtricyc03.m
%  and re-packaging its output in order to always output 
%  haltk as an M-by-1 vector and Haltk as
%  an M-by-(3+2*M) matrix, if called for.  Any rows associated with
%  non-existent measurements are arbitrarily set to zero in
%  both haltk and Haltk.  This strategy ensures that the EKF, UKF,
%  and BSEKF algorithm software that calls this measurement
%  model will properly account for the actual lack of measurements
%  in these cases.  Note that a particle filter will have to
%  be modified in this case to eliminate the re-sampling if all rows
%  of haltk are identically zero independent of the xk input. 
%  The necessary modifications can be accomplished by
%  not resampling after a measurement update if the particle 
%  weights are all very close to each other after the update.
%

%
%  Determine the number of merry-go-rounds.
%
   M = size(imykflaghist,2);
%
%  Set up zero-valued or empty initial output arrays.
%
   haltk = zeros(M,1);
   if iflaghonly == 1
      Haltk = [];
   else
      Haltk = zeros(M,(3+2*M));
   end
%
%  Call hfunct_blindtricyc03.m in order to get the measurement
%  model vector and, if called for, its Jacobian in the 
%  non-zero-padded format.
%
   if nargin > 8
      [hk,Hk,mykvec] = blindtricyc03_hfunct(xk,k,iflaghonly,br,...
                                            Xvec,Yvec,rhovec,...
                                            imykflaghist,yaltkhist);
   else
      [hk,Hk,mykvec] = blindtricyc03_hfunct(xk,k,iflaghonly,br,...
                                            Xvec,Yvec,rhovec,...
                                            imykflaghist);
   end
%
%  Assign the proper elements of hk to haltk and, if called for,
%  the proper elements of Hk to Haltk.
%
   lk = size(mykvec,1);
   if lk > 0
      haltk(mykvec,1) = hk;
      if iflaghonly == 0
         Haltk(mykvec,:) = Hk;
      end
   end