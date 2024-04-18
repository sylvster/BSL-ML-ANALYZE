## Getting Started

### Installation

A conda environment is recommended for downloading all the required libraries.

```
conda env create -f environment.yml
```

```
conda env create -f environment.yml -n NAME
```

Alternatively, you may manually download the following list of libraries to run the code:

```
pandas 2.1.1
numpy 1.24.3
pytorch 2.1.1
matplotlib 3.8.2
```

### Usage

To use the BSL Machine Learning model, simply place a TAR file containing the station data into the same directory as analyze.py is in, and then run

```
python3 analyze.py [TAR FILE NAME]
```

As an example, a TAR file from the CLRV station was attached into this repository. To analyze its contents, you can run

```
python3 analyze.py CLRV.BK.HHE.00.PDF.tar
```