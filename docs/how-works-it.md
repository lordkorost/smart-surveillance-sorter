# 🧠 How It Works

## The Problem
Gli nvr hanno uno spazio limitato dipendente dalla grandezza dell'hdd, quindi se impostati per registrare 24/7 esauriscono lo spazio e cancellano i video dei giorni precedenti.
In caso di guasti, mi è capitato in due diverse occasioni, smettono di registrare e non si hanno le prove dell'intrusione. Ho risolto impostanto l'NVR per registrare solo gli eventi
sul mio home server tramite ftp. Il vento, una ragnatela, la pioggia, nonostante le impostazioni alla sensibilità e le maschere per ignorare alcune zone continuano a registrare su ftp
molti video "inutili" e guardarli tutti quando serve non è comodo. La soluzione è un sistema che categorizza i video, estrae i frame per avere un feedback immediato di cosa contiene
il video, cosi da poter trovare subito i video "interessanti" per la sicurezza, passando dal dover guardare 400+ video ad una decina per avere la certezza che il mio sistema non abbia
categorizzato male eventi rilevanti. Questo è esattamente quello che fa Smart Surveillance Sorter.

## The Idea
L'idea è quella di esaminare i video senza doverli guardare tutti realmente ed eliminare quelli che non interessano prima di riempire 1 terabyte di registrazioni su ftp ed evitare 
che un guasto al hdd dell'nvr mi faccia perdere una reale intrusione. Il sistema però dev'essere abbastanza veloce per essere sostenibile. Inoltre dev'essere un sistema altamente
personalizzabile, se cambio una telecamera o l'nvr o aggiungo telecamere da interno (per ora ho solo telecamere da esterno) il sistema deve continuare a funzionare e ordinare i video
correttamente. Questo è reso possibile dalla configurazione di quasi tutti i paramentri per singola telecamera, un po' come funziona sul nvr dove si possono impostare eventi trigger e 
confidenze minime per telecamera. La velocità di esecuzione e la precisione del sorter dipende da due macro fattori, l'hardware del pc e le impostazioni del sorter, però le impostazioni
del nvr sono egualmente importanti, più video "falsi" registra, piu video il sorter dovrà esaminare, più il tempo di elaborazione aumenta e anche i falsi positivi.

## The Pipeline
Ho ideato due pipeline con l'idea di una cpu friedly e una più precisa ma gpu dipendente per avere tempi utili. Entrambe usano come primo passaggio YOLO.
Funziona così:
1) Yolo esamina le eventuali immagini salvate dall'nvr associate ad un video, cerca solo persone e se l'immagine contiene una persona con confidenza alta, quel video viene già catalogato come persona (questo è un passaggio velocissimo nell'ordine di secondi e permette di non dover analizzare quei video nel secondo passaggio)
2) Yolo esamina i video che non sono stati categorizzati nel primo step. I video vengono esaminati in base ai settaggi per telecamera, alla velocità dipendente dagli fps reali del video. Ad esempio con Stride 0.6 sec, viene analizzato 1 frame ogni 0.6 secondi, saltando stride * FPS frame tra un'analisi e l'altra. Nei frame vengono ricercati e salvati, in base alla modalità e alle ignore labels specifiche per telecamera (persone,animali,veicoli).

Qua la pipeline si divide:
un modo veloce:
3a) Blip-Clip: i frame e i relativi crop salvati da Yolo vengono inviati e processati da blip-clip che decidono effettivamente la categoria del video. Opzionale: Per una maggiore precisione si può fare un ulteriore passaggio, si passano i video "incerti" dove yolo e blip-clip non concordano a vision che esamina i frame e decide la categoria per quei pochi video rimanenti.
un modo più lento e preciso
3b) Vision: i frame di yolo vengono mandati ad un modello vision tramite ollama (qwen3-vl:8b si è dimostrato il più affidabile) e decide la categoria del video.

Per non avere falsi negativi sulle persone e accettare pochi falsi positivi il sistema in caso di risposta yolo persona con confidenza "media-alta" e blip-clip o vision risposta diversa, si crede a yolo.
I passaggi di blip-clip o vision servono principalmente a ridurre i falsi positivi.


## Why a Chain?
La sequenza dei passaggi non è casuale, ma come ripeto tutto nasce dalle impostazioni del NVR.
Con un NVR ben configurato tutti i successivi passaggi sono più veloci e precisi.
Yolo sulle immagini nvr riesce a ridurre i video da esaminare in maniera molto veloce (secondi)
Yolo sui video è il passaggio che influenza di più in termini di tempo, meno video "inutili" l'NVR ha registrato , meno video da esaminare.
Blip-Clip su frame e crop salvati da yolo è un passaggio veloce dipendente dalla modalità e relativo numero di frame salvati
Vision sui frame salvati da yolo è un passaggio più lento, dipendente sempre dal numero di frame salvati

## Person First — Always
In ambito sicurezza le persone hanno la priorità assoluta al costo di qualche falso positivo. 
Il sistema è tarato in modo da favorire le persone in ogni passaggio, con boost, threshold bassi etc. questo si riflette in una quasi totale assenza di falsi negativi sulle persone
anche in casi dove nello stesso video c'è sia un veicolo che una persona o una persona e un animale. 
(circa 10 su piu di 7000 video esaminati durante i test dopo aver affinato la configurazione per singola telecamera),quei 10 sono stati comunque recuperati dai video delle altre telecamere, 
in un buon sistema di videosorveglianza è buona norma avere le telecamere ad "incrociarsi", quello che viene perso su una telecamera viene presa dall'altra che riprende 
la stessa scena ma dall'angolazione opposta.

## Early Exit
Il meccanismo che ho ideato per velocizzare il passaggio di Yolo sui video, oltre ad uno stride impostabile, è il meccanismo che permette a yolo di non esaminare l'intero video se rileva una persona o un animale. Specificato un numero di occorrenze e un time gap in secondi, yolo smette di analizzare un video e passa al prossimo quando nello stesso video viene rilevato un animale o una persona numero occorrenze volte a distanza di almeno time gap secondi. Ad esempio con i valori num_occ 3 e time_gap_sec 3, in un video di 1 minuto entra in scena al secondo 5 una persona e rimane nell'inquadratura, al secondo 8 (seconda occorrenza a distanza di 3 secondi), al secondo 11(terza occorrenza a distanza di 3 secondi) yolo ha concluso e passa al prossimo video. con solo 11 secondi analizzati su 1 minuto di video. La stessa cosa con gli animali, animale secondo 5, animale secondo 10, animale secondo 21, yolo ha finito e passa al prossimo video.

Sui veicoli l'early exit non si applica. Il meccanismo serve ad esaminare più velocemente i video che contengono effettivamente persone o animali, come per un'analisi reale, se il video non contiene quello che cerchiamo per scoprire che non c'è dobbiamo vederlo tutto. 

Il ragionamento è che se c'è una persona allora è sicuro smettere di esaminare il video. Se c'è un animale di solito c'è una persona che lo accompagna se è un animale domestico, quindi la persona viene rilevata correttamente, mentre se è un animale randagio quasi sicuramente non ci sarà accanto una persona, il gatto o il cane dovrebbe scappare se ci fosse una persona (rischio calcolato). Per i veicoli questo non si può applicare, i veicoli non si muovono in autonomia (si spera) quindi è altamente probabile che scenda una persona e se scattasse il meccanismo di early exit sul veicolo la persona sarebbe persa e si avrebbe un falso negativo sulle persone, cosa che voglio evitare a tutti i costi.

Questo meccanismo di early exit permette di esaminare in minor tempo i video che contengono qualcosa di utile per la sicurezza. I video vuoti registrati dal NVR per "errore" verranno esaminati interamente come farebbe una persona reale e qui di nuovo entra in gioco la corretta impostazione del NVR e della pulizia delle telecamere per avere meno video da esaminare, un processo più veloce.

## YOLO vs BLIP vs Vision
Yolo: (yolov8l): con le giuste confidenze minime si è dimostrato molto affidabile sulle persone, anche sagome lontane, al buio, ma tende a trovare persone anche con confidenza alta in situazioni di ombre, pioggia o altro.
Blip e Clip:  con i giusti fake keywords riesce a limitare i falsi positivi sulle persone di yolo, ma sugli animali ha qualche "problema" non riesce bene a distinguerli
Vision (qwen3-vl:8b): con il giusto prompt è un vero descrittore, analizza i frame (4k) non perdendo quasi niente (in alcuni casi "difficili" ci sono thinking molto lunghi che rallentano il processo) ma è molto affidabile sugli animali, riduce drasticamente sia i falsi positivi sulle persone che i falsi negativi sugli animali. Alcune limitazioni, con temperature 1 il modello non risponde sempre la stessa cosa sulla stessa immagine (casi molto ambigui) quindi può "generare" dei falsi negativi sulle persone quando sono veramente lontane e piccole (frame 4k) al buio nonostante yolo le abbia trovate ma con confidenza bassa da non far scattare la clausola di salvaguardia. In generale migliora la precisione rispetto a blip e clip su animali e persone. 

## Known Limitations
I gatti muovendosi veloce soprattutto di notte non vengono sempre rilevati da yolo oppure vengono scartati dal passaggio blip-clip
Se yolo esamina i video con stride impostati alti per ridurre i tempi di esecuzione, persone che corrono potrebbero essere perse (casi rari)
Persone parzialmente visibili come ad esempio una testa nell'angolino inferiore dell'inquadratura, un braccio o un viso attraverso una finestra o il finestrino dell'auto, alcune volte vengono rilevate da yolo, altre volte no. Non lo ritengo un problema per la sicurezza, inoltre con telecamere ben posizionate (ad incrociare) la testa nell'angolo, nel video dell'altra telecamera risulta una figura completa.
Il sistema essendo tarato per dare la massima importanza alle persone può categorizzare animali come persone se in uno dei passaggi (yolo o blip) l'animale viene scambiato per una persona.