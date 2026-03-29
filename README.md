# End-to-End Low-Latency ML System for Loan Approval

Ten projekt prezentuje w pełni zautomatyzowany, produkcyjny potok uczenia maszynowego (ML Pipeline) służący do oceny zdolności kredytowej. Głównym celem systemu jest zaserwowanie modelu głębokiej sieci neuronowej (MLP), który został poddany rygorystycznej, **dwuetapowej kompresji matematycznej**, redukując jego rozmiar z ~63 KB do niespełna 19 KB, przy jednoczesnym zachowaniu niemal 90% dokładności (Accuracy).

System został zaprojektowany w architekturze mikroserwisowej (FastAPI + Docker) z naciskiem na minimalne opóźnienia (Low-Latency) i wdrażanie na urządzeniach o ograniczonych zasobach (Edge AI / CPU-only inference).

---

## 🏗️ Architektura Systemu i Modelu Bazowego

System opiera się na wielowarstwowym perceptronie (MLP) zbudowanym w **PyTorch Lightning**. Aby zapobiec przeuczeniu i ustabilizować proces treningu, architektura wykorzystuje warstwy *Batch Normalization* oraz *Dropout*.



Cały cykl życia modelu zarządzany jest w architekturze MLOps:
1. **DVC (Data Version Control):** Śledzenie wersji surowych zbiorów danych.
2. **MLflow Model Registry:** Rejestrowanie eksperymentów, hiperparametrów oraz wersjonowanie artefaktów (skalery, wagi modeli).
3. **Automated Pipeline:** Oskryptowany proces (od pobrania danych, przez trening, po kompresję).
4. **FastAPI & Docker:** Serwowanie skwantyzowanego modelu jako bezstanowego punktu końcowego API.

---

##  Matematyczna Optymalizacja Modelu

To, co wyróżnia ten system, to odejście od heurystycznego "zgadywania" na rzecz optymalizacji opartej na dowodach matematycznych i teorii gier. Proces dzieli się na trzy fazy:

### Faza 1: Feature Pruning (Explainable AI / SHAP)
Przed ingerencją w wagi sieci, redukujemy redundancję na wejściu. Traktując predykcję jako grę kooperacyjną, obliczamy wartości Shapleya (SHAP) dla każdej cechy. Cechy o znikomym wpływie na wynik decyzyjny ($\mathbb{E}[|\phi_i|] \approx 0$) są na stałe usuwane z potoku.



### Faza 2: Structural Pruning (First-Order Taylor Expansion)
Zamiast maskować wagi (co nie zmniejsza zużycia RAMu), system fizycznie usuwa najmniej istotne neurony z warstw ukrytych. Ważność neuronu oceniana jest na podstawie aproksymacji rozwinięcia w szereg Taylora pierwszego rzędu dla funkcji straty $\mathcal{L}$. Wynik (Score) dla neuronu $i$ w warstwie $l$ wyraża się wzorem:

$$S_i^{(l)} \approx \left| \frac{\partial \mathcal{L}}{\partial a_i^{(l)}} \cdot a_i^{(l)} \right|$$

Dzięki temu usuwane są tylko te struktury ($a_i$), których brak wygeneruje najmniejszy gradient błędu. Przynosi to **kwadratową redukcję** liczby operacji zmiennoprzecinkowych (FLOPs).

### Faza 3: Dynamiczna Kwantyzacja (Int8)
Ostatnim krokiem jest mapowanie precyzji wag z 32-bitowych zmiennoprzecinkowych ($r \in \mathbb{R}$) na 8-bitowe liczby całkowite ($q \in \mathbb{Z}$) za pomocą transformacji afinicznej, wykorzystującej współczynnik skali ($S$) i punkt zerowy ($Z$):

$$q = \text{round}\left( \frac{r}{S} + Z \right)$$

Umożliwia to błyskawiczną inferencję procesorom (CPU), które są znacznie wydajniejsze w operacjach na liczbach całkowitych.

---

##  Wyniki i Trade-off



Zastosowanie optymalizacji strukturalnej (usunięcie ~30% neuronów) oraz kwantyzacji poskutkowały redukcją rozmiaru modelu o współczynnik **3.35x**. 

| Wersja Modelu | Rozmiar na dysku | Accuracy | F1 Score |
| :--- | :---: | :---: | :---: |
| Baseline (FP32) | 63.13 KB | 0.912 | 0.920 |
| Pruned (FP32) | 35.94 KB | 0.897 | 0.906 |
| **Quantized (Int8)** | **18.83 KB** | **0.897** | **0.906** |

Spadek dokładności na poziomie zaledwie ~1.5% jest akceptowalnym kosztem biznesowym w zamian za potężny spadek opóźnień sieci (latency) oraz kosztów utrzymania infrastruktury serwerowej.

---

##  Stack Technologiczny

* **Deep Learning:** PyTorch, PyTorch Lightning
* **Optymalizacja & XAI:** SHAP, torch.quantization, torch.nn.utils.prune
* **MLOps / Tracking:** MLflow, DVC (Data Version Control)
* **API / Backend:** FastAPI, Uvicorn, Pydantic
* **Infrastruktura:** Docker, Docker Compose
* **Data Science:** Pandas, Scikit-Learn, NumPy

---

## Jak uruchomić system (Quickstart)

Projekt jest gotowy do wdrożenia w środowisku skonteneryzowanym. Nie wymaga lokalnej instalacji bibliotek Pythonowych poza silnikiem Docker.

1. **Sklonuj repozytorium:**
   ```bash
   git clone [https://github.com/JakubDras/loan-approval-system.git](https://github.com/JakubDras/loan-approval-system.git)
   cd loan-approval-system