function y = getPDF_PF( xp, tblx)

    N       =   length( xp ) ; 
    dx      =   tblx(2)-tblx(1) ;
    ndata   =   length( tblx ) ; 
    pdf     =   zeros( ndata, 1 ) ; 
    area    =   0.0 ;

    for i = 1 : ndata-1
       
        bin1=   tblx( i ) ; 
        bin2=   tblx(i+1) ; 
        
        for j = 1 : N
            
            if ( bin1 <= xp(j) ) && ( xp(j) < bin2 )                      % Assign pdf
                        
                pdf(i) =   pdf(i) + 1 ;
                        
            end
            
        end
        
        area   =   area + pdf(i) * dx ; 
        
    end
    
    pdf         =   pdf / area ; 
    
    y           =   pdf ; 
