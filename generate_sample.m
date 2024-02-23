function [X,Y,change_points,bw,mIndex,T,Smi,Sleng] = generate_sample(...
          GMW,SegmCount,lo_length,hi_length,NSI)
%% Обеспечивает наличие всех классов и в обучении и в контроле!      
%% GMW - matrix of generating models weights M x K
%% SegmCount - number of quasi-stationary segments (of randomn length)
%% lo_length,hi_length - min and max length of QS segment
%% X,Y - output matrix
%% change_points
%% bw - полуширина размаха отклика вокруг 0
%% mIndex индекс модели для текущей точки 
%% Поочередно для всех SegmCount сегментов
%% Генерируем случайно их длины, индексы моделей, сигналы Xs,
%%    отклики Ys, заполняем change_points и mIndex
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
mIndex = []; % Номера генерирующих моделей для всех T точек выборки
[M,K] = size(GMW); % Число моделей и размерность входных переменных
if SegmCount<M     % Число сегментов не должно быть меньше числа моделей !
    error('Small number of segments!');
end

% Формируем последовательность генерирующих моделей
mi1 = (1:M);                       % Каждая по одному разу  
mi2 = randi([1,M],1,SegmCount-M);  % Выбрали случайно повторы
Smi = [mi1 mi2];
p = randperm(length(Smi));        % Случайная перестановка       
Smi = Smi(p);

Sleng = randi([lo_length,hi_length],1,SegmCount);  % Segment Length's
change_points = cumsum([1 Sleng]);
T = sum(Sleng);

for s=1:SegmCount
    cModel = GMW(Smi(s),:);
    
    %Ys = randn(Sleng(s),1)*10/NSI; % Шум добавили заранее !!!
    Ys = randn(Sleng(s),1)/NSI; % Шум добавили заранее !!!
    Xs = randn(Sleng(s),K);
    
    Ys = Ys+Xs*cModel';
    X = [X; Xs];                     % Достраиваем матрицу X
    Y = [Y; Ys];                     % Достраиваем столбец Y
    Is = ones(Sleng(s),1)*Smi(s);       % Достраиваем столбец mIndex
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

