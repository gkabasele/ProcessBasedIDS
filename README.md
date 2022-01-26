# A process-aware Time-Based IDS
Repository for the paper on the Time-based IDS. An IDS designed to detect process-oriented attack targeting ICS/SCADA. The IDS learned temporal properties of the physical process to perform its detection.

There are three IDS implemented that can be found in the ids directory:
  - The Time-based IDS where the main function can be found in ids/timeChecker.py
  - The Invariant-based IDS which can be found in the ids/idsInvariants.py
  - The Prediction-based IDS (using Autoregression) which can be found in ids/idsAR.py

## ICS Simulated 
![alt text](https://github.com/gkabasele/ProcessBasedIDS/raw/master/physical_process_smaller.png)

The simulated process traces used for the experiment are available [here](https://drive.google.com/drive/folders/1Tz1srP7S6Fasr2JTQOyZBHVGrKlFz2Td?usp=sharing)

