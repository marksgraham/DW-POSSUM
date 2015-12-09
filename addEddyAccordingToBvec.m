function [ new_pulse ] = addEddyAccordingToBvec(tint, delta, Delta, Gdiff, ep, tau, bval, bvecx,bvecy,bvecz  )
%ADDEDDYACCORDINGTOBVEC simulates the effects of eddy currents in a PGSE sequence and adds them to a pulse
%sequence matrix obtained from POSSUM. It calculates the amount of eddy
%distoriton to add based on a given gradient strength and bvec.
%
% INPUTS: 
%          tint        - time of the first diffusion pulse
%          delta       - pulse length / s
%          Delta       - diffusion time /s
%          Gdiff       - strength of diffsion gradients
%          ep          - strength of eddy currents as a fraction of Gdiff
%          tau        - eddy decay time
%          bvecx,y,z   - bvec in x, y and z directions

%Extract pulse and puplse info - these should be in this directory
pulse=read_pulse('pulse');
pulseinfo=load('pulse.info');

%Quick hack to ensure eddy currents scale with b-value
Gdiff=Gdiff*bval/2000;

%Extract time
time = pulse(:,1)';
numSlices = pulseinfo(13);
TRslice = pulseinfo(4);
RFtime = time(8); %Time reserved for RF pulses

Eddyx=zeros(4,length(pulse));
Eddyy=zeros(4,length(pulse));
Eddyz=zeros(4,length(pulse));

for i = 0:numSlices-1
    t(1)=tint+TRslice*i;
    t(2)= tint+delta+TRslice*i;
    t(3)=tint+Delta+TRslice*i;
    t(4)=tint+delta+Delta+TRslice*i;
    RF=RFtime+TRslice*i;
    
    %Remember the signs needsto change
    for j=1:4
        addx = (ep * Gdiff * bvecx * ( exp(-(time - t(j)) /tau ))).* (time>t(j)).*(time>RF);
        addy = (ep * Gdiff * bvecy * ( exp(-(time - t(j)) /tau ))).* (time>t(j)).*(time>RF);
        addz = (ep * Gdiff * bvecz * ( exp(-(time - t(j)) /tau ))).* (time>t(j)).*(time>RF);
        
        addx(isnan(addx))=0;
        addy(isnan(addy))=0;
        addz(isnan(addz))=0;
        
        if j ==2 || j==3
            addx=addx*-1;
            addy=addy*-1;
            addz=addz*-1;
        end
        
        Eddyx(j,:) = Eddyx(j,:)+addx;
        Eddyy(j,:) = Eddyy(j,:)+addy;
        Eddyz(j,:) = Eddyz(j,:)+addz;
    end
    
   % Eddyx(1,:) = Eddyx(1,:) + (ep * Gdiff * bvecx * ( exp(-(time - t1) /tau ))).* (time>t1).*(time>RF);
    %Eddyx(2,:) = Eddyx(2,:) - (ep * Gdiff * bvecx * ( exp(-(time - t2) /tau ))).* (time>t2).*(time>RF);
    %Eddyx(3,:) = Eddyx(3,:) - (ep * Gdiff * bvecx * ( exp(-(time - t3) /tau ))).* (time>t3).*(time>RF);
    %Eddyx(4,:) = Eddyx(4,:) + (ep * Gdiff * bvecx * ( exp(-(time - t4) /tau ))).* (time>t4).*(time>RF);
    
%     Eddyy(1,:) = Eddyy(1,:) + (ep * Gdiff * bvecy * ( exp(-(time - t1) /tau ))).* (time>t1).*(time>RF);
%     Eddyy(2,:) = Eddyy(2,:) - (ep * Gdiff * bvecy * ( exp(-(time - t2) /tau ))).* (time>t2).*(time>RF);
%     Eddyy(3,:) = Eddyy(3,:) - (ep * Gdiff * bvecy * ( exp(-(time - t3) /tau ))).* (time>t3).*(time>RF);
%     Eddyy(4,:) = Eddyy(4,:) + (ep * Gdiff * bvecy * ( exp(-(time - t4) /tau ))).* (time>t4).*(time>RF);
%     
%     Eddyz(1,:) = Eddyz(1,:) + (ep * Gdiff * bvecz * ( exp(-(time - t1) /tau ))).* (time>t1).*(time>RF);
%     Eddyz(2,:) = Eddyz(2,:) - (ep * Gdiff * bvecz * ( exp(-(time - t2) /tau ))).* (time>t2).*(time>RF);
%     Eddyz(3,:) = Eddyz(3,:) - (ep * Gdiff * bvecz * ( exp(-(time - t3) /tau ))).* (time>t3).*(time>RF);
%     Eddyz(4,:) = Eddyz(4,:) + (ep * Gdiff * bvecz * ( exp(-(time - t4) /tau ))).* (time>t4).*(time>RF);
    
end
%Set eddys to zero in the first 7 time points, reserved for RF pulses
%for i=1:4
%    Eddyx(i,1:7)=0;
%    Eddyy(i,1:7)=0;
%    Eddyz(i,1:7)=0;
%end
    
new_pulse=pulse;
%Add to approptiate channels
new_pulse(:,6)=pulse(:,6)+sum(Eddyx,1)';
new_pulse(:,7)=pulse(:,7)+sum(Eddyy,1)';
new_pulse(:,8)=pulse(:,8)+sum(Eddyz,1)';


write_pulse('pulse_new',new_pulse,1);
end