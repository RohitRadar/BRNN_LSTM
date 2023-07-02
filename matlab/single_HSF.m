function P = single_HSF(teta,fi)
%#codegen


bit=1;
c=299792458;
f=8.65e9;
lambda=c/f;
k = 2*pi/lambda;        
%fi=80;
%teta=40;

n=bit;
E=2*pi/2^n*linspace(0,2^n-1,2^n); %whats this?
        phiR=(fi-180)*pi/180;
        thetaR=(teta)*pi/180;
        du=lambda/4;
        DN=6*lambda;  %whats this? Number of unit cells in a row or column
        DM=DN;
        dsx = du*2*pi*cos(phiR)*sin(thetaR)/lambda;   %incident angle is zero so other term is zero
        dsy = du*2*pi*sin(phiR)*sin(thetaR)/lambda;   %incident angle is zero so other term is zero
        M=round(DM/du);
        N=round(DN/du);
        B=zeros(M,N);
        C=B;
        D=C;
        for i=1:M
            for j=1:N
                B(i,j) =((i*dsx) + (j*dsy));       
                C(i,j) = B(i,j);

                while B(i,j)>=2*pi
                    B(i,j)=B(i,j)-2*pi;
                end
                while B(i,j)<0
                    B(i,j)=B(i,j)+2*pi;
                end
                D(i,j)=B(i,j);
                [q, p]=min(abs(E-abs(B(i,j))));
                B(i,j)=E(p);
                if B(i,j)==2*pi
                B(i,j)=0;
                end
            end
        end
       dG=du;Ps=179;Ts=181;
        phi=linspace(-1,180,Ps);
        theta=linspace(-1,91,Ts);
        new = zeros(181, 179);
        F = complex(new, 0); %what is this
        PH = zeros(M,N);
        for a=1:M
            for b=1:N
                PH(a,b)=B(a,b);
                phase = exp(-1j*(PH(a,b) + k*dG*(a-0.5).*sin(theta*pi/180)'*cos(phi*pi/180) + k*dG*(b-0.5).*sin(theta*pi/180)'*sin(phi*pi/180)));
                
                F = F + phase;
                
            end
        end
        
        test= abs(F);
        test_max = max(max(test));
        test_norm = test/test_max;
       % check = max(max(test_norm))
        r_phi=80;
        r_theta = 50;
        
       
        
        
        
        [m1,n1]= (min(abs(theta - r_theta))); %what is r_theta
        [m2,n2]= (min(abs(phi - r_phi)));  
        P = test_norm(n1, n2);
        error= teta - r_theta;
        
%         surf(PH)
%         view(0,90)
%         xlim([1 M])
%         ylim([1 N])        
%     
%         figure,
%         surf(phi,theta,abs(F).^2)
%         view(0,90)
%         xlim([0 180])
%         ylim([0 90])
