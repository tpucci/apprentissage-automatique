%  -------------------------------------------------
% | Q-Learning                                      |
% | Adaptation de l'exemple de cours                |
% | -                                               |
% | Trouver le meilleur chemin vers la pièce 5.     |
%  -------------------------------------------------


% Énoncé : Initialisation de la matrice R
% ---
R = -1*ones(6);
doors = [[0,4]; [4,3]; [4,5]; [2,3]; [1,3]; [1,5]];
wins = [[1,5];[4,5];[5,5]];

for i = 1:size(doors,1) % Création des portes
    R(doors(i,1)+1,doors(i,2)+1) = 0;
    R(doors(i,2)+1,doors(i,1)+1) = 0;
end

for i = 1:size(wins,1) % Chemins gagnants
    R(wins(i,1)+1,wins(i,2)+1) = 100;
end

% Configuration
% ---
alpha = 1;
gamma = .8;
nEpisodes = 100;

% Initialisation de la matrice Q
% ---
Q = zeros(size(R));

% Début de l'algorithme
% ---
randomStates = randi([1 size(R,2)],1,100);

for i = 1:nEpisodes % Boucle de nEpisodes
    beginningState = randomStates(i);
    Q = qLearn(Q,R,alpha,gamma,6, 5+1); % Appel de la fonction récursive qLearn
end

QNormalized = round(Q./max(max(round(Q)))*100) % Affichage du résultat arrondi