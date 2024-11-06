function J = lorenz4DJ(~)
syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 F

% Define the system of ODEs
ODEs = [
        x40*(x2  - x39) - x1  + F;
        x1 *(x3  - x40) - x2  + F;
        x2 *(x4  - x1 ) - x3  + F;
        x3 *(x5  - x2 ) - x4  + F;
        x4 *(x6  - x3 ) - x5  + F;
        x5 *(x7  - x4 ) - x6  + F;
        x6 *(x8  - x5 ) - x7  + F;
        x7 *(x9  - x6 ) - x8  + F;
        x8 *(x10 - x7 ) - x9  + F;
        x9 *(x11 - x8 ) - x10 + F;
        x10*(x12 - x9 ) - x11 + F;
        x11*(x13 - x10) - x12 + F;
        x12*(x14 - x11) - x13 + F;
        x13*(x15 - x12) - x14 + F;
        x14*(x16 - x13) - x15 + F;
        x15*(x17 - x14) - x16 + F;
        x16*(x18 - x15) - x17 + F;
        x17*(x19 - x16) - x18 + F;
        x18*(x20 - x17) - x19 + F;
        x19*(x21 - x18) - x20 + F;
        x20*(x22 - x19) - x21 + F;
        x21*(x23 - x20) - x22 + F;
        x22*(x24 - x21) - x23 + F;
        x23*(x25 - x22) - x24 + F;
        x24*(x26 - x23) - x25 + F;
        x25*(x27 - x24) - x26 + F;
        x26*(x28 - x25) - x27 + F;
        x27*(x29 - x26) - x28 + F;
        x28*(x30 - x27) - x29 + F;
        x29*(x31 - x28) - x30 + F;
        x30*(x32 - x29) - x31 + F;
        x31*(x33 - x30) - x32 + F;
        x32*(x34 - x31) - x33 + F;
        x33*(x35 - x32) - x34 + F;
        x34*(x36 - x33) - x35 + F;
        x35*(x37 - x34) - x36 + F;
        x36*(x38 - x35) - x37 + F;
        x37*(x39 - x36) - x38 + F;
        x38*(x40 - x37) - x39 + F;
        x39*(x1  - x38) - x40 + F
        ];

% Extract variables
x = [x1;  x2;  x3;  x4;  x5;  x6;  x7;  x8;  x9;  x10; ...
     x11; x12; x13; x14; x15; x16; x17; x18; x19; x20; ...
     x21; x22; x23; x24; x25; x26; x27; x28; x29; x30; ...
     x31; x32; x33; x34; x35; x36; x37; x38; x39; x40];

% Compute the Jacobian matrix
J = jacobian(ODEs, x);
