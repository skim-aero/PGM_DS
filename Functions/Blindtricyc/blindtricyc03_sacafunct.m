function [sa,ca,dsadx,dcadx,d2sadx2,d2cadx2] = blindtricyc03_sacafunct(x,iflagsacadervs)
%
%  Copyright (c) 2011 Mark L. Psiaki.  All rights reserved.  
% 
%  This function computes
%                      _
%                     |  sin(x)/x   if x ~= 0
%            sa(x) = -|
%                     |_ 1          if x == 0
%
%  and
%                      _
%                     |  (cos(x) - 1)/x   if x ~= 0
%            ca(x) = -|
%                     |_ 0                if x == 0
%  
%  and their first 2 derivatives with respect to x.
%
%  If iflagsacadervs == 0, then only sa and ca are calculated.
%  If iflagsacadervs == 1, then only sa, ca, dsadx, and dcadx are
%  computed.  Otherwise, all 6 outputs are computed.
%

%
%  This clause does the case of only sa and ca.  The different
%  iflagsacadervs cases are split up into 3 independent clauses
%  in hopes of computing faster in the case of not computing
%  all of the outputs.
%
   if iflagsacadervs == 0
%
%  Use a series approximations for sin(x) and for cos(x) if x is small.
%
      if abs(x) <= 0.02
         xsq = x^2;
         sa = 1;
         ca = 1;
         jjfac0 = 12;
         jjfac1 = 11;
         jjfac2 = 10;
         for jj = 1:5
            denfac = 1/(jjfac1*jjfac2);
            sa = 1 - (sa*xsq)*denfac;
            denfacb = 1/(jjfac0*jjfac1);
            ca = 1 - (ca*xsq)*denfacb;
            jjfac0 = jjfac0 - 2;
            jjfac1 = jjfac1 - 2;
            jjfac2 = jjfac2 - 2;
         end
         ca = -0.5*x*ca;
      else
         sinx = sin(x);
         cosx = cos(x);
         oneox = 1/x;
         sa = sinx*oneox;
         ca = (cosx - 1)*oneox;
      end
      dsadx = [];
      dcadx = [];
      d2sadx2 = [];
      d2cadx2 = [];
%
%  This clause does the case of sa and ca and their first
%  derivatives only.
%       
   elseif iflagsacadervs == 1
%
%  Use a series approximations for sin(x) and for cos(x) if x is small.
%
      if abs(x) <= 0.02
         xsq = x^2;
         twox = 2*x;
         dsadx = 0;
         sa = 1;
         dcadx = 0;
         ca = 1;
         jjfac0 = 12;
         jjfac1 = 11;
         jjfac2 = 10;
         for jj = 1:5
            denfac = 1/(jjfac1*jjfac2);
            dsadx = - (dsadx*xsq + sa*twox)*denfac;
            sa = 1 - (sa*xsq)*denfac;
            denfacb = 1/(jjfac0*jjfac1);
            dcadx = -(dcadx*xsq + ca*twox)*denfacb;
            ca = 1 - (ca*xsq)*denfacb;
            jjfac0 = jjfac0 - 2;
            jjfac1 = jjfac1 - 2;
            jjfac2 = jjfac2 - 2;
         end
         dcadx = -0.5*(ca + x*dcadx);
         ca = -0.5*x*ca;
      else
         sinx = sin(x);
         cosx = cos(x);
         oneox = 1/x;
         sa = sinx*oneox;
         dsadx = (cosx - sa)*oneox;
         ca = (cosx - 1)*oneox;
         dcadx = -(sinx + ca)*oneox;
      end
      d2sadx2 = [];
      d2cadx2 = [];
%
%  This clause does the case of sa and ca and both of their
%  derivatives.
%       
   else
%
%  Use a series approximations for sin(x) and for cos(x) if x is small.
%
      if abs(x) <= 0.02
         xsq = x^2;
         twox = 2*x;
         d2sadx2 = 0;
         dsadx = 0;
         sa = 1;
         d2cadx2 = 0;
         dcadx = 0;
         ca = 1;
         jjfac0 = 12;
         jjfac1 = 11;
         jjfac2 = 10;
         for jj = 1:5
            denfac = 1/(jjfac1*jjfac2);
            d2sadx2 = - (d2sadx2*xsq + 2*dsadx*twox + sa*2)*denfac;
            dsadx = - (dsadx*xsq + sa*twox)*denfac;
            sa = 1 - (sa*xsq)*denfac;
            denfacb = 1/(jjfac0*jjfac1);
            d2cadx2 = -(d2cadx2*xsq + 2*dcadx*twox + ca*2)*denfacb;
            dcadx = -(dcadx*xsq + ca*twox)*denfacb;
            ca = 1 - (ca*xsq)*denfacb;
            jjfac0 = jjfac0 - 2;
            jjfac1 = jjfac1 - 2;
            jjfac2 = jjfac2 - 2;
         end
         d2cadx2 = -0.5*(2*dcadx + x*d2cadx2);
         dcadx = -0.5*(ca + x*dcadx);
         ca = -0.5*x*ca;
      else
         sinx = sin(x);
         cosx = cos(x);
         oneox = 1/x;
         sa = sinx*oneox;
         dsadx = (cosx - sa)*oneox;
         d2sadx2 = (-sinx - 2*dsadx)*oneox;
         ca = (cosx - 1)*oneox;
         dcadx = -(sinx + ca)*oneox;
         d2cadx2 = -(cosx + 2*dcadx)*oneox;
      end
   end