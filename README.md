**Problematic Internet Use**

Description

Questo progetto mira a [descrizione degli obiettivi principali], utilizzando tecniche di machine learning e analisi dei dati per [scopo specifico, come classificare o etichettare dati mancanti].

**Struttura delle Cartelle**
data/raw/: Contiene i dati grezzi originali.
data/processed/: (Opzionale) Contiene i dati preprocessati pronti per l'uso.
models/: Contiene i modelli addestrati e ottimizzati.
src/: Codice sorgente del progetto, suddiviso in più moduli:
clean_data.py: Funzioni per la pulizia dei dati.
preprocessing.py: Script di preprocessing, inclusa la standardizzazione e l'imputazione dei valori mancanti.
unsupervised_labelling.py: Implementazione di modelli di clustering per etichettare le righe senza valori nella colonna sii.
statistical_analysis.py: Funzioni per l'analisi statistica e la riduzione dimensionale (PCA).
main.py: Script principale che coordina l'intero flusso di lavoro.
notebooks/: (Opzionale) Notebook Jupyter per l’esplorazione iniziale dei dati e lo sviluppo.
reports/: (Opzionale) Contiene report e visualizzazioni generate dal progetto.


**Dipendenze**
Lista delle principali dipendenze del progetto:

Python >= 3.8
pandas
scikit-learn
matplotlib
seaborn
Assicurati di installare le dipendenze:

pip install -r requirements.txt
Esecuzione del Progetto
Preprocessing dei Dati:
Esegui main.py per caricare i dati e preprocessarli.
Addestramento del Modello e Etichettatura:
unsupervised_labelling.py contiene la logica per applicare l’unsupervised learning ai dati senza etichetta e assegnare le pseudo-etichette.
Analisi Statistica:
statistical_analysis.py usa PCA per identificare le variabili più influenti e visualizza la matrice di correlazione delle variabili.

**Esecuzione dei Test**

Per assicurarti che la pipeline funzioni correttamente senza errori, puoi eseguire i test automatici inclusi nel progetto.

Prerequisiti
Assicurati di essere nella directory principale del progetto e di avere installato tutte le dipendenze.

Comando per Eseguire i Test
Apri il terminale nella directory principale del progetto e digita il seguente comando per eseguire tutti i test:

python3 -m unittest discover -s tests
Questo comando:

Avvia unittest come modulo principale (-m unittest).
Usa discover per trovare automaticamente i file di test nella directory specificata (-s tests).
Verifica che il file test_pipeline.py e altri file di test nella cartella tests vengano eseguiti.
Se i test passano senza errori, vedrai un messaggio simile a:

----------------------------------------------------------------------
Ran X tests in Y.YYYs

OK
Se invece uno o più test falliscono, unittest mostrerà l'errore e la parte del codice che ha generato il problema.


**Esempio di Utilizzo**
python src/main.py
Note Importanti
Gestione del Dataset: Assicurati di collocare i dati grezzi nella cartella data/raw/.
Modelli Salvati: I modelli addestrati saranno salvati nella cartella models/.

**Contatti**
Per domande o suggerimenti, contattare [tuo indirizzo email].

