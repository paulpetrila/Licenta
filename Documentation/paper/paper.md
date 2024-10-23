# Introducere

 In ultimii ani, detectarea emoțiilor în text a devenit din ce în ce mai populară
	datorită potențialului vast și aplicațiilor pe care îl are în marketing,
	psihologie, inteligență artificială etc. Accesul larg la baze de date mari de date tip text,
	mai ales texte cu opinii și sentimente puternice,
	a făcut posibilă antrenarea de modele de limbaj din ce în ce mai performante și credibile.
	De asemenea, posibilitatea generării de imagini folosind texte prin intermediul modelelor precum DALL-E [@dalle] a deschis accesul către această paradigmă pentru milioane de oameni.
  Această inovație a permis utilizatorilor să exprime concepte și idei vizuale în mod creativ, transformând cuvintele în imagini complexe și captivante. Astfel, prin intermediul modelelor precum DALL-E, oamenii pot explora noi dimensiuni ale creației artistice și pot adăuga un nivel de expresivitate și accesibilitate în comunicarea lor vizuală. Este evidentă contribuția semnificativă pe care aceste modele o aduc în diversificarea și democratizarea creației digitale.
	

Detectarea emoțiilor în lingvistica computațională este procesul de identificare a emoțiilor 
discrete exprimate în text. Analiza emoțiilor poate fi privită ca o evoluție naturală a analizei 
sentimentelor și a modelului său mai detaliat. 

Domeniul inteligenței artificiale generative, bazat pe modele de limbaj de mari dimensiuni 
(Large Language Models - LLMs) a avut parte de schimbări mari în ultimii ani. 
LLM-urile au devenit suficient de avansate încât pot fi folosite în aplicații de complexitate diferită.


## Formularea Problemei:

Obiectivul acestei cercetări este de a dezvolta o metodă eficientă pentru generarea de imagini bazate pe categoriile exprimate în texte, folosind Rețele Neuronale Adversariale (GANs) și tehnici avansate de interpretare a limbajului natural.

Problema principală pe care o adresăm este legată de coerența și fidelitatea imaginilor generate în raport cu conținutul textelor de intrare. În timp ce abordări anterioare s-au concentrat pe generarea de imagini bazate pe baza unor categorii sau cuvinte cheie, această cercetare vizează specific interpretarea și reprezentarea vizuală a categoriilor exprimate în limbaj.



<!-- Astfel, prin metodele descrise mai în detaliu mai jos, am creat un sistem care:

1. Procesează la intrare un text

2. Îl clasează într-o categorie de emoție

3. Trimite răspunsul mai departe ca intrarea la generatorul C-GAN

4. Afișează o imagine generată pe baza emoției transmise -->



# Concepte de bază în Învățarea Automată și Inteligența Artificială

Întrucât Învățarea Automată și Inteligența Artificială reprezintă domenii în plină dezvoltare, cu aplicații din ce în ce mai diverse în viața noastră cotidiană, vom explora câteva dintre conceptele de bază specifice.

## Metode de Învățare Automată

Învățarea Automată reprezintă o ramură a Inteligenței Artificiale care se ocupă cu dezvoltarea și utilizarea algoritmilor capabili să învețe modele din date și să facă predicții fără a fi programate explicit. Printre metodele de bază ale Învățării Automate se numără Clasificarea, Regresia și Clusterizarea.

### Clasificare

Clasificarea este o tehnică de Învățare Automată supervizată utilizată atunci când se dorește încadrarea unui element într-una dintre categorii prestabilite. Modelul este instruit pe un set de date etichetat, învățând să facă distincții și să generalizeze aceste distincții pentru a clasifica corect elemente noi. Exemple de aplicații includ recunoașterea facială, diagnosticul medical și filtrarea spamului în e-mailuri.

### Regresie

Regresia este, de asemenea, o tehnică de Învățare Automată supervizată, dar, spre deosebire de clasificare, are ca obiectiv prezicerea unei valori continue în loc de încadrarea în categorii discrete. Algoritmii de regresie analizează relațiile dintre variabilele de intrare și produc o funcție care aproximează valorile de ieșire dorite. Această tehnică este des utilizată în predicția prețurilor, analiza de trenduri și prognozele economice.

### Clusterizare

Clusterizarea face parte din categoria tehnologiilor de Învățare Automată nesupervizată. Scopul principal al acestei metode este gruparea datelor similare în cluster-uri sau grupuri distincte. Algoritmii de clusterizare identifică tipare în date fără a avea informații prealabile despre categoriile acestora. Aplicațiile includ segmentarea pieței, analiza social media și organizarea automată a bazelor de date.

## Rețele Neurale Artificiale (ANNs)

Rețelele neuronale reprezintă o paradigmă de învățare a mașinilor de calcul inspirată de structura și funcționarea sistemelor neuronale biologice. Acestea includ în straturi de unități numite neuroni, conectate între ele prin conexiuni caracterizate prin ponderi. Prin antrenament, aceste ponderi sunt ajustate pentru a permite rețelei să învețe relații complexe și modele din datele de intrare.

Un aspect crucial al rețelelor neuronale este capacitatea lor de învățare a reprezentărilor ierarhice și abstracte. Într-o rețea neurală tipică, avem un strat de intrare care primește datele inițiale, straturi ascunse care procesează aceste date prin aplicarea unor transformări matematice, și un strat de ieșire care furnizează rezultatele dorite. Fiecare conexiune între neuroni are o pondere asociată (weight), și învățarea constă în ajustarea acestor ponderi pentru a minimiza o funcție de cost, astfel încât rețeaua să producă rezultatele dorite.

Procesul de învățare se realizează prin propagarea înapoi a erorii (backpropagation) și optimizarea funcției de cost folosind algoritmi precum gradientul descendent (gradient descent). În esență, rețeaua își ajustează intern parametrii pentru a reduce discrepanța între predicțiile sale și valorile de ieșire dorite, indicate în datele de antrenament.

Există diverse arhitecturi de rețele neuronale adaptate pentru diferite sarcini. De exemplu, rețelele feedforward sunt cele mai simple, cu informația transmisă într-o singură direcție, de la intrare la ieșire. Rețelele recurente (RNN) introduc elementul temporal, fiind capabile să proceseze secvențe de date. Rețelele convoluționale (CNN) sunt specializate în prelucrarea datelor spațiale, cum ar fi imagini.

Un alt concept important este cel al funcțiilor de activare, care introduc non-linearități în rețea, permitând acesteia să învețe relații complexe. Funcții precum ReLU (Rectified Linear Unit) sunt frecvent utilizate pentru aceasta.

Cu trecerea timpului, rețelele neurale au devenit tot mai adânci și mai complexe, conducând la apariția rețelelor neuronale profunde (DNN). Un exemplu de modele profunde îl oferă transformerele. Acestea au avut un impact semnificativ în rezolvarea sarcinilor complexe precum recunoașterea obiectelor, traducerea automată și generarea de conținut.


Rețelele neuronale reprezintă o paradigmă puternică în domeniul inteligenței artificiale, fiind capabilă să învețe și să reprezinte informații complexe. Aceste sisteme au contribuit semnificativ la progresele din ultimii ani în domeniul înțelegerii automate a datelor.


## Rețele Neurale de Convoluție (CNNs)

Rețelele Neurole de Convoluție (CNNs) sunt specializate în prelucrarea datelor spațiale, cum ar fi imagini. Ele folosesc straturi de convoluție pentru a detecta caracteristici locale și pentru a reduce treptat dimensiunile datelor. Aceste rețele sunt eficiente în recunoașterea de modele și ierarhizarea caracteristicilor, permitând învățarea automată a acestora. CNN-urile sunt larg utilizate în domenii precum recunoașterea de obiecte, segmentarea imaginilor și analiza imaginilor medicale.

## Rețele Neurale Adversariale Generative (GAN)

O Rețea Neurală Adversariale Generativă (GAN) constă din două rețele neurale distincte: un generator și un discriminator. Generatorul încearcă să creeze date noi care să fie dificil de distins de datele reale, în timp ce discriminatorul încearcă să facă distincția între datele reale și cele generate. Aceste două rețele sunt antrenate în mod adversar, îmbunătățindu-se reciproc. GAN-urile sunt folosite în generarea de conținut nou, cum ar fi imagini realiste, text sau chiar sunete.

## Rețele Neurale Adversariale Generative Condiționale (C-GAN)

Rețelele Neurale Adversariale Generative Condiționale (C-GAN) extind conceptul de GAN prin adăugarea unui element condițional. Astfel, generatorul și discriminatorul primesc, în plus față de datele de intrare aleatorii, și informații suplimentare care pot controla sau direcționa procesul de generare. Acest control condițional îmbunătățește capacitatea de generare și permite utilizatorilor să influențeze rezultatele dorite. C-GAN-urile sunt utilizate în diverse domenii, precum generarea de imagini de peisaje, editarea imaginilor și în alte aplicații creative.



![Arhitectura unui Transformer[@vaswani2023attention]](/home/paul/Coding/Licenta/paper/assets/Transformer.png)




## BERT

![Arhitectura BERT[@devlin-etal-2019-bert]](/home/paul/Coding/Licenta/paper/assets/BERT_ARCH.png)

BERT, sau Bidirectional Encoder Representations from Transformers, reprezintă o inovație semnificativă în domeniul preprocesării limbajului natural și în înțelegerea contextului semantic al cuvintelor în texte. Dezvoltat de către Google AI, BERT a fost introdus în 2018 și a avut un impact deosebit asupra sarcinilor legate de procesarea limbajului natural (NLP).

La baza BERT se află arhitectura Transformer, care a fost prezentată inițial în [@vaswani2023attention]. Arhitectura tip Transformer a introdus un mecanism de atenție, care permite modelului să se concentreze asupra anumitor părți ale intrării, ceea ce a dus la performanțe cu mult îmbunătățite în comparație cu arhitecturile anterioare.

BERT îmbunătățește această arhitectură prin abordarea problemei direcționale a modelelor anterioare. În loc să proceseze textul într-o singură direcție (de la stânga la dreapta sau de la dreapta la stânga), BERT utilizează o abordare bidirecțională, adică analizează contextul atât înainte, cât și în urma cuvântului curent. Această caracteristică bidirecțională îi permite să captureze relațiile semantice complexe și să înțeleagă mai bine contextul global al propozițiilor.

Un aspect cheie al BERT este pre-antrenarea. Înainte de a fi folosit pentru o anumită sarcină, modelul este antrenat pe o cantitate masivă de date text fără etichete, dezvoltând o înțelegere profundă a limbajului natural. În timpul acestei pre-antrenări, BERT învață să prezică cuvintele lipsă din contextul lor, creând reprezentări semantice bogate pentru fiecare cuvânt.

După pre-antrenare, BERT poate trece prin procesul de ”Fine Tuning” pentru sarcini specifice, cum ar fi clasificarea de texte pentru analiza emoțiilor. Fine-tuning-ul implică ajustarea parametrilor modelului pe un set de date etichetate pentru a se adapta la sarcina specifică.

Prin abordarea bidirecțională, pre-antrenarea detaliată și fine-tuning adaptat, BERT a obținut rezultate remarcabile într-o serie de benchmark-uri pentru NLP, depășind alte modele existente și stabilind noi standarde în înțelegerea contextului semantic în texte, de exemplu în detectarea de știri false [@azizah2023performance]. Impactul său extins se reflectă în utilizarea sa pe scară largă în aplicații precum motoare de căutare, asistenți virtuali și alte sisteme bazate pe limbaj natural.


## Analiza emoțiilor dintr-un text

Analiza emoțiilor dintr-un text reprezintă o sarcină importantă în domeniul procesării limbajului natural (NLP), iar rețelele neuronale, în special modelele precum BERT, au avut un impact substanțial în îmbunătățirea performanțelor acestor sarcini. În cadrul analizei emoțiilor, obiectivul este de a atribui etichete specifice unui text care exprimă anumite stări emoționale, cum ar fi bucurie, tristețe, furie sau frică.

BERT aduce contribuții semnificative în abordarea acestei sarcini, datorită înțelegerii sale profunde a contextului semantic al cuvintelor și frazelor. Capacitatea sa bidirecțională de a analiza contextul atât înainte, cât și în urma cuvântului curent se dovedește crucială în înțelegerea subtilităților semantice care definesc emoțiile în limbaj.

În etapa de pre-antrenare, BERT învață să recunoască și să captureze nuanțele subtile ale limbajului natural, inclusiv expresiile asociate cu anumite emoții. Modelul este expus la o vastă cantitate de texte variate și învață să prezică cuvintele lipsă în contextul lor, dezvoltând reprezentări semantice bogate pentru fiecare termen.

Fine-tuning-ul reprezintă etapa prin care BERT este adaptat pentru sarcina specifică de analiză a emoțiilor. Modelul este antrenat pe un set de date etichetate, unde textele sunt asociate cu categoriile corespunzătoare de emoții. Ajustarea parametrilor se realizează pentru a optimiza performanța modelului în identificarea corectă a emoțiilor.

Utilizarea BERT în analiza emoțiilor dintr-un text a adus beneficii semnificative, iar rezultatele obținute au depășit cu mult performanțele modelelor anterioare. Acest lucru se datorează, în mare parte, abordării sale inovatoare, care integrează înțelegerea contextuală profundă cu tehnicile de pre-antrenare și fine-tuning.


--- 

# Metodologie de lucru

## Generarea imaginilor

În cadrul acestei cercetări, am adoptat o abordare interdisciplinară, combinând cunoștințe din domeniul rețelelor neuronale adversariale (GANs), interpretării limbajului natural (NLP) și generării de imagini. Am început prin a explora literatura existentă referitoare la modelele GAN și tehnici de interpretare a limbajului natural pentru a identifica metodele și abordările relevante.

Pasul următor a constat în proiectarea și implementarea unei arhitecturi GAN adaptate pentru generarea de imagini. Am ales să utilizăm următoarele variante ale rețelei, testate pe mai multe seturi de date.  



În primă fază, am antrenat rețeaua pe un set de date format din 5 categorii de flori, pentru a putea observa comportarea acesteia la generarea formelor naturale, precum petalele. 

![Poză cu florile generate](/home/paul/Coding/Licenta/paper/assets/florigenerate.png)

Apoi, am testat flexibilitatea acesteia, oferindu-i un set de date variat, cu mai multe clase din domenii fără legătură.

![Imagini generate de rețea, cu un set de date mai vast](/home/paul/Coding/Licenta/paper/assets/ExemplePoza.png)




<!-- ## Generarea imaginilor folosind C-GAN[@mirza2014conditional] -->

<!-- ![Exemplu arhitectura DALLE[@dalle]](/home/paul/Coding/Licenta/paper/assets/DALLE_ARCH.png) -->






Structura rețelelor adversare folosită este prezentată în figura:

 ![Arhitectura Generatorului în stânga și a Discriminatorului în dreapta](/home/paul/Coding/Licenta/paper/assets/GeneratorSiDiscriminator.png)


Setul de date folosit la antrenare este oferit de unsplash lite^[https://github.com/unsplash/datasets?tab=readme-ov-file], din care am luat doar 4 clase: (winter, autumn, spring, summer).



## Identificarea emoțiilor

Antrenarea BERT a urmărit următoarele etape:



### Alegerea setului de date și preprocesarea lui 

Preprocesarea datelor, cu următoarele etape:

- Normalizare: Această etapă constă în conversia textului într-o formă standard sau comună. De exemplu, poate include etapa de capitalizare uniformă a textului, reducerea acestuia doar la litere mici sau mari.

- Stemming: Acesta este procesul de reducere a cuvintelor la forma lor de bază sau rădăcină prin eliminarea sufixelor. De exemplu, "running", "runs" și "ran" pot fi reduse la "run". Stemming ajută la reducerea numărului de cuvinte în text și simplificarea vocabularului.

- Lematizare: Această etapă constă în reducerea cuvintelor la forma lor canonică sau de dicționar, luând în considerare partea lor de vorbire și contextul. De exemplu, "is", "are" și "were" pot fi reduse la "be". Lematizarea este similară cu stemming, dar este mai precisă și mai sofisticată.

- Eliminarea cuvintelor de legătură: Aceasta este etapa de eliminare a cuvintelor care sunt foarte comune și nu adaugă mult sens sau informație textului. De exemplu, "the", "a", "and" etc. Eliminarea cuvintelor de legătură ajută la reducerea zgomotului și dimensiunii textului și se concentrează pe cuvintele importante.

- Eliminarea semnelor de punctuație: Aceasta este etapa de eliminare a semnelor de punctuație din text, cum ar fi virgule, puncte, semne de întrebare etc. Eliminarea semnelor de punctuație ajută la eliminarea simbolurilor inutile și face textul mai curat și mai simplu.

- Corectarea ortografiei și greșelilor gramaticale

- Tokenizare: Această etapă reprezintă procesul de descompunere a textului în unități mai mici numite token-uri. Token-urile pot fi cuvinte, propoziții, paragrafe etc. Tokenizarea ajută la împărțirea textului în segmente semnificative care pot fi ușor procesate de modelele de procesare a limbajului natural (NLP).

### Alegerea Modelului Pre-antrenat BERT

Procesul de selecție a modelului pre-antrenat BERT a implicat analiza atentă a mai multor factori, având în vedere importanța alegerii potrivite în contextul sarcinii specifice pe care o abordăm. Decizia de a utiliza modelul bert-base-uncased a fost susținută de mai multe considerente.

#### Dimensiunea și Complexitatea Modelului

Bert-base-uncased reprezintă o variantă a modelului BERT care oferă un echilibru între performanță și dimensiune. Având o arhitectură robustă, este adecvat pentru sarcinile de procesare a limbajului natural, acoperind o gamă variată de contexte semantice.

#### Disponibilitatea Resurselor Computaționale

Alegerea unui model mai mic, cum ar fi bert-base-uncased, a avut în vedere restricțiile resurselor computaționale disponibile pentru antrenarea și implementarea ulterioară a modelului. Această variantă permite eficiența în procesul de fine-tuning, fără a compromite semnificativ performanța.

#### Performanța Generală

Modelul bert-base-uncased a fost pre-antrenat pe o cantitate semnificativă de date din limba engleză și a obținut rezultate bune pe o serie de benchmark-uri pentru diverse sarcini de procesare a limbajului natural. Alegerea sa se bazează pe performanța generală în contextul sarcinii noastre specifice.

#### Compatibilitatea cu Problema Noastră

Dat fiind faptul că problema noastră implică recunoașterea emoțiilor în texte, am evaluat capacitatea modelului bert-base-uncased de a captura nuanțele subtile ale limbajului și de a înțelege contextul emoțional.



### Fine-Tuning

În etapa de fine-tuning, am parcurs următoarele sub-etape pentru a adapta modelul la sarcina specifică:

#### Definirea Setului de Date de Antrenament și Validare
    
Am identificat și definit un set de date de antrenament care să cuprindă o varietate reprezentativă a datelor pentru sarcina dată. De asemenea, am separat un set de date de validare pentru a evalua performanța modelului într-un mediu independent.

#### Antrenarea modelului pe setul de date de antrenament

Am inițiat procesul de antrenare, unde modelul a fost expus la setul de date de antrenament definit. Am utilizat funcția de cost și optimizatorul pentru a ajusta ponderile modelului în timpul iterațiilor, facilitând asimilarea cunoștințelor din datele de antrenament.

#### Evaluarea pe setul de evaluare

După finalizarea etapei de antrenare, am evaluat performanța modelului pe setul de date de validare. Acest pas ne-a furnizat o măsură a generalizării modelului și a calității predicțiilor sale în fața datelor noi și nevăzute.


Prin aceste sub-etape ale procesului de fine-tuning, am căutat să optimizăm performanța modelului nostru pentru sarcina specifică, asigurându-ne că acesta poate face predicții precise și robuste în fața unor date variate.


# Planificare Activității Viitoare

În efortul continuu de a aduce contribuții semnificative în domeniul recunoașterii emoțiilor, lucrarea curentă urmează a fi extinsă în diverse direcții de cercetare și dezvoltare. Următoarele subiecte reprezintă direcții importante pentru abordarea și explorarea în continuare a domeniului:



<!-- 
Pentru dezvoltarile viitoare sunt avute in vedre urmatorele directii importante:

A. Proiectarea unui CGAN pentru generarea de imagini care indica anotimpuri, folsoind setul d edate indicat  in...
B. Proietarea sistemului de clasificare a textului bazat pe BERT pentru setul de date indicat in..., care include farze ce decriu diefrite anitimpuri, fara folsoirea directa a nuemlor acestora
C. Interfatarea celor doua sisteme
D. Interfatarea siatemului de clasificare atextului cu un sistem speech2text

 -->

## Proiectarea unui CGAN[@goodfellow2014generative] pentru generarea de imagini care indică anotimpuri

## Proiectarea sistemului de clasificare a textului bazat pe BERT pentru setul de date creat de mine ^[https://github.com/Pauwul/SeasonsTextDataset], care include fraze ce descriu diferite anotimpuri, fără folosirea directă a numelor acestora.

## Interfațarea celor două sisteme


## Interfațarea sistemului de clasificare a textului cu un sistem speech2text

Ca parte a extinderii funcționalității, se va explora implementarea unui model Speech to Text. Acesta va oferi utilizatorilor posibilitatea de a comunica mai simplu cu aplicația, având capacitatea de a transforma discursul în text și de a integra aceste informații în procesul de analiză a emoțiilor.


## Detalierea Teoriei Aplicate în Modelul Creat

Lucrarea va fi extinsă pentru a include o descriere mai detaliată a teoriei aplicate în modelul dezvoltat. Această secțiune va furniza un cadru conceptual robust pentru înțelegerea fundamentelor modelului și a deciziilor luate în procesul de dezvoltare.

## Interfață de Comunicare și Gestionare a Fotografiilor

O atenție deosebită va fi acordată dezvoltării unei interfețe de comunicare îmbunătățite cu aplicația, facilitând gestionarea și vizualizarea fotografiilor create. Această îmbunătățire a experienței utilizatorului este esențială pentru implementarea cu succes a aplicației.


## Utilizarea unor Metrici de Evaluare

Pentru cuantificarea calității imaginilor generate, care să ia în calcul cât sunt de realiste acestea, nu doar asemănarea cu imaginile dorite indicate în setul de antrenare. 

## Comunicarea între Matlab și Python:

Integrarea și comunicarea între medii de dezvoltare diferite, cum ar fi Matlab și Python, a reprezentat o provocare semnificativă. Soluțiile aplicate pentru facilitarea acestui proces vor fi detaliate, explorând metodele eficiente de sincronizare și schimb de date între aceste două platforme.

# Provocări întâmpinate

În cursul dezvoltării acestui proiect, au fost identificate diverse provocări care au necesitat soluții inovatoare și strategii de gestionare. Aceste provocări, însoțite de soluțiile și direcțiile de dezvoltare adoptate, sunt prezentate în continuare:

## Gestionarea Dependențelor în Python

Una dintre provocările întâmpinate a fost gestionarea dependințelor în limbajul de programare Python, o sarcină adesea considerată dificilă. Pentru a depăși această problemă, s-a optat pentru utilizarea unei imagini Docker, oferind astfel o soluție eficientă pentru a asigura consistența și reproducibilitatea mediului de dezvoltare.

<!-- 
## Gestionarea Eficientă a Timpului în Timpul Semestrului:

Menținerea unui echilibru între cerințele studențești și gestionarea vieții personale a fost o provocare esențială. În această secțiune, se vor discuta strategiile adoptate pentru a gestiona eficient timpul pe durata semestrului, asigurându-se că progresul proiectului rămâne constant și sustenabil. -->


## Duratele lungi de antrenare a modelelor:

Un alt aspect critic al proiectului a fost gestionarea duratelor lungi de antrenare a modelelor.

Aceste provocări, identificate și depășite pe parcursul proiectului, au fost abordate cu o abordare proactivă și creativă. Astfel, am avut oportunitatea de a mă informa mai in detaliu despre toate etapele dezvoltării unei soluții software, dar cu precădere asupra celor care folosesc inteligența artificială.

# Concluzii

Această lucrare investighează domenii variate, de la generarea de imagini prin intermediul rețelelor C-GAN până la analiza textelor folosind modele precum BERT. Prin schimbarea setului de date, am pus în evidență generalitatea rețelelor GAN, iar aplicabilitatea modelelor precum BERT reiese din ușurința prin care acesta poate să se plieze pe necesitatea utilizatorului, prin procesul de fine-tuning.


# References

