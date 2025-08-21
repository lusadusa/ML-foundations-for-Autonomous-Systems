@echo off
setlocal enableextensions

REM === Directory del repo = cartella dove sta questo .bat ===
set "REPO=%~dp0"

REM === Attiva l'env (percorso standard Anaconda sotto LocalAppData) ===
call "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" activate ml-foundations

REM === (Fallback) se sopra non esiste, prova nella home ===
if not "%CONDA_PREFIX%"=="" goto :env_ok
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate ml-foundations
:env_ok

REM === Aggiungi le DLL (SDL2 ecc.) al PATH + variabili anti-crash ===
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"

REM === Vai nel repo e avvia JupyterLab (fallback al notebook se manca) ===
cd /d "%REPO%"
where jupyter-lab >nul 2>&1 && (
  jupyter lab
) || (
  jupyter notebook
)

echo.
echo [INFO] Chiuso Jupyter. Premi un tasto per uscire.
pause
endlocal
