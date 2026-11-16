# Schur-Batching auf warp9, 16.11.2026

Mit **16 Threads und `--update-block 128`** sinkt die mediane
Faktorisierungszeit gegenüber der Standardbreite 32 um **21,0 %**.
Gegenüber MKL Pardiso benötigt diese Einstellung **33,6 % weniger Zeit**.
Die Verbesserung ist eine Änderung der Batchbreite im vorhandenen Solver.

| Konfiguration | Faktorisierung | Lösen | Faktorisierung + Lösen |
|---|---:|---:|---:|
| FEMultiFrontal, Batch 32 | 7,405 s | 0,418 s | 7,823 s |
| FEMultiFrontal, Batch 128 | **5,851 s** | 0,464 s | **6,315 s** |
| MKL Pardiso, SPD | 8,816 s | 0,725 s | 9,509 s |

Alle Zahlen sind Mediane aus fünf Messungen nach einem verworfenen Warm-up.
Die letzte Spalte ist der Median der jeweils gemeinsam gemessenen Zeiten.
Faktorisierungszeiten über die fünf Messungen: Batch 32 **7,119–8,980 s**,
Batch 128 **5,499–6,889 s**, MKL **8,498–9,322 s**.

Gemessen wurde ausschließlich auf **warp9.sla.rwth-aachen.de**, AMD EPYC 7713,
mit denselben 16 gebundenen CPUs: `0,1,2,4,5,9,10,11,12,13,14,15,17,18,19,20`.
Die Maschine wird gemeinsam genutzt; die Wiederholungen dokumentieren die
auftretende Streuung. `OMP_WAIT_POLICY=ACTIVE`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, dynamische Teamgrößen aus. Der native Treiber bestätigt
`threads_used=16` in jeder Messung.

Referenzcommit: `50f1d8802ee8fce4cbcc22f914074fa21685653e`.
GCC 14.4.0, `-O3 -march=native -DNDEBUG -std=c++20 -fopenmp`;
MKL **2026.1.0**. Beide nativen Einstellungen benutzen denselben ausführbaren
Treiber, dieselbe Permutation und denselben symbolischen Plan. Die Reihenfolge
der Batchbreiten wechselt zwischen den Wiederholungen. MKL benutzt seine eigene
Ordnung; seine symbolische Analyse wird einmal ausgeführt, anschließend werden
frische numerische Faktorisierungen mit Phase 22 und Lösungen mit Phase 33 gemessen.

Die Tabelle misst die numerischen Phasen. FEM-Assembly, Matrixexport, Import,
Ordering und symbolische Analyse liegen außerhalb dieser Zeiten. Der native
Aufbau aus Ordering, Permutation und symbolischer Analyse dauerte im
Bestätigungslauf 3,189 s; MKLs Phase 11 dauerte 3,005 s. Die vorgelagerte
Indexkonvertierung des MKL-Treibers ist darin nicht enthalten.

Das Modell ist ein einseitig eingespannter Volumenbalken aus **304.704 C3D8**:
576 × 23 × 23 Elemente, E=210000, nu=0,3, 997.056 gesamte und
**995.328 freie Freiheitsgrade**. FEMaster auf warp9 erzeugte die reduzierte
Steifigkeitsmatrix `CANTILEVER_1M_A.mtx`; beide Solver lesen denselben daraus
erzeugten CSC-Cache mit 38.541.072 gespeicherten Einträgen im unteren Dreieck.
Der native Faktor enthält 1.010.480.940 Einträge, die größte Front hat 5.328 Zeilen.

Beide Treiber verwenden dieselbe künstliche, nichtkonstante exakte Lösung und
prüfen den Fehler am ursprünglichen Operator außerhalb der Zeitmessung.
Maximale relative Residuen: Batch 32 **8,46e-16**, Batch 128 **8,72e-16**,
MKL **4,56e-16**. Maximale relative Lösungsfehler liegen zwischen **2,40e-10**
und **3,67e-10**. Die vorhandenen Korrektheits- und Robustheitstests wurden auf
warp9 sowohl für die Referenz als auch für die Cache-Blockungsvariante erfolgreich
ausgeführt (`failures=0`).

Der Vorteil größerer Batches passt zum geringeren wiederholten Speicherverkehr
für die gepackte Beitragsmatrix. Gemessen ist die vollständige Faktorisierung;
parallele, überlappende Schur-Aufrufzeiten wurden nicht zu einer vermeintlichen
Wall-Time-Phase aufsummiert.

Weitere geprüfte Ansätze:

- Cache-Streifen und Begrenzung der gleichzeitig berührten Ausgabezeilen:
  überwiegend langsamere Schur-Mikrokerne, nicht übernommen.
- Gepackte 4×12- statt 6×8-Registerkachel: kein robuster Vorteil bei großen Fronten.
- Ein oder zwei statt vier OpenMP-Jobs pro Thread: überwiegend langsamer,
  insbesondere mit 16 Threads; Ausgaben bitgleich zur Referenz.
- Mehr Separatorsuchläufe, andere Balancegrenze und größere Grobgraphen:
  symbolisch geprüft. Die beste Variante reduzierte die geschätzte Arbeit nur
  um etwa 1,2 %, andere erhöhten sie; keine Änderung am Produktions-Ordering.
- Batchbreiten 32, 64, 128 und 256: auf diesem Modell mit 16 Threads war 128
  im Pilotlauf am schnellsten. Bei vier Threads lagen die Zeiten der Breiten
  wesentlich näher zusammen. Deshalb bleibt die Empfehlung an diesen Test und
  diese Threadzahl gebunden; der globale Standard wurde nicht geändert.

Die Server-Arbeitskopien, Matrizen, ausführbaren Dateien und vollständigen
Protokolle liegen unter:

```text
/tmp/femultifrontal-schur-20261116-f_eggers/
```

Wiederholen des entscheidenden Vergleichs auf warp9:

```sh
cd /tmp/femultifrontal-schur-20261116-f_eggers
python benchmarks/confirm_warp9.py "$PWD" --threads 16 --repeat 6
```

Für den normalen Solver lautet die relevante Einstellung:

```sh
OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_PLACES=cores \
  ./baseline/build/native-omp/mfsolve model/CANTILEVER_1M_A.mtx \
  --threads 16 --update-block 128 --repeat 6
```

Der Treiber des Bestätigungslaufs bindet zusätzlich die oben genannten CPUs.
Matrix und Protokolle liegen auf der separaten Server-Partition `/tmp`, da die
Systempartition voll war. Die vorhandenen Repositories unter `~/libs/` wurden
nicht verändert.

Lokal gesicherte Daten:
[Zusammenfassung](results/warp9_20261116/summary.csv),
[native Einzelmessungen](results/warp9_20261116/native.log),
[MKL-Einzelmessungen](results/warp9_20261116/mkl.log),
[Umgebung und Matrix-SHA256](results/warp9_20261116/environment.json).
Diese Daten sind Bestandteil des Repositories. Weitere Versuchsprotokolle und
Quellvarianten liegen in der lokalen Sicherung unter `dev/warp9_20261116`;
die großen Matrizen bleiben auf warp9. Die Benchmark-README beschreibt außerdem
den Aufbau und Aufruf aus einem frischen Checkout.
