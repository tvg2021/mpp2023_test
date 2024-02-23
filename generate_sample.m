function [X,Y,change_points,bw,mIndex,T,Smi,Sleng] = generate_sample(...
          GMW,SegmCount,lo_length,hi_length,NSI)
%% ������������ ������� ���� ������� � � �������� � � ��������!      
%% GMW - matrix of generating models weights M x K
%% SegmCount - number of quasi-stationary segments (of randomn length)
%% lo_length,hi_length - min and max length of QS segment
%% X,Y - output matrix
%% change_points
%% bw - ���������� ������� ������� ������ 0
%% mIndex ������ ������ ��� ������� ����� 
%% ���������� ��� ���� SegmCount ���������
%% ���������� �������� �� �����, ������� �������, ������� Xs,
%%    ������� Ys, ��������� change_points � mIndex
%% GMW = randn(M,K)
if nargin == 0
    GMW = randn(5,20);
    M=5;
    SegmCount = 9;
    lo_length = 60;
    hi_length = 470;
    NSI = 1;
end

X = [];
Y = [];
mIndex = []; % ������ ������������ ������� ��� ���� T ����� �������
[M,K] = size(GMW); % ����� ������� � ����������� ������� ����������
if SegmCount<M     % ����� ��������� �� ������ ���� ������ ����� ������� !
    error('Small number of segments!');
end

% ��������� ������������������ ������������ �������
mi1 = (1:M);                       % ������ �� ������ ����  
mi2 = randi([1,M],1,SegmCount-M);  % ������� �������� �������
Smi = [mi1 mi2];
p = randperm(length(Smi));        % ��������� ������������       
Smi = Smi(p);

Sleng = randi([lo_length,hi_length],1,SegmCount);  % Segment Length's
change_points = cumsum([1 Sleng]);
T = sum(Sleng);

for s=1:SegmCount
    cModel = GMW(Smi(s),:);
    
    %Ys = randn(Sleng(s),1)*10/NSI; % ��� �������� ������� !!!
    Ys = randn(Sleng(s),1)/NSI; % ��� �������� ������� !!!
    Xs = randn(Sleng(s),K);
    
    Ys = Ys+Xs*cModel';
    X = [X; Xs];                     % ����������� ������� X
    Y = [Y; Ys];                     % ����������� ������� Y
    Is = ones(Sleng(s),1)*Smi(s);       % ����������� ������� mIndex
    mIndex = [mIndex;Is];
end  
a = min(Y);
b = max(Y);
bw = max(abs(a),abs(b));
q=1;
mean(Y)
std(Y)
mean(X)
std(X)
% figure; plot(mIndex); ylim([0 M+0.5]); grid on

