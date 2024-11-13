"""
Esecuzione del Test della Pipeline

Per eseguire questo test, apri un terminale nella directory principale del progetto e usa il comando:

    python3 -m unittest discover -s tests

Questo comando eseguirà tutti i test definiti in questo file e nelle altre parti della cartella 'tests'.
"""

import sys
import os

# Aggiungi la directory principale del progetto al percorso
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from src import main

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Setup per i test, se necessario
        pass

    def test_main_execution(self):
        """Testa che main.py termini senza errori."""
        try:
            main.main()
        except Exception as e:
            self.fail(f"Il main ha generato un'eccezione inaspettata: {e}")

if __name__ == "__main__":
    unittest.main()
