Hi I am Sritej Kumbar. Thanks for dropping by as I would love to contribute to you via my passion for engineering via this GitHub. I am a Master's student researcher in C++/ Simulation in Mechanical Engineering at TU Darmstadt . This is Worlds 326th Ranked University and I have worked for SAFRAN Aerospace which is Worlds 2nd Largest Aerospace Company. This is some Collection of Code/Projects where I utilised Modern C++ for Solver Developement & PyTorch for Machine Learning Application in Production at my home institute. Please open each directories for detailed explanation. Some codes are contributed by Users . Please honor them at their LinkedIn or Email due to their hardwork if you utilise their code.

Short Brief:


1. ActiveCarSpoilerCFDSim (Colab Project India-India)

This is done as Tutorial in Numerical Simulation Methods.
The takeaway of this project was Aerodynamic Study, that determined angle for Right Downforce at Cornering and Breaking for Active Car Spoiler used by Mazda's GT Sports car model, due to Optimisation significance in F1 Car Aerodynamics by use of CFD . Mesh Size was 16Million Cells refined at Wall and k-Omega SST turb. Model utilised. CAD Modelling & Simulation in STARCCM+ was done by me. Akhil Unikrishnan supported in Documenting and Market Research.

Fahrzeugaerodynamik/ Car Aerodynamics by Prof. Schutz at BMW Munchen,DE lecture helped additionally.


Structure

|--Design (CAD Files)
|--Photos for the Project (Contain Simulation Images)
|--Team 4 Active Car Spoiler by Sritej and Akhil..... (Documentation)


2. Brusselator Solver

The takeaway of the project is that this simulation solver studies oscillatory behavior in chemical systems. It serves as a simplified representation of complex reactions the Belousov-Zhabotinsky reaction , which exhibits periodic changes in chemical concentrations over time. One important focus was that traditional C++ elements were utilized and a lot was achieved by ensuring correct basics. Basics are important.

This is solo project

Structure

|--brusselator.cpp (Declares ODE)
|--integrator.cpp (Contain integrator RK4 scheme)
|--PT4SC.pdf (Contain theory of Solver)
|--solver.cpp. (Solves ODE)
|--CMaleLists.txt (This is how you run the code)

3. EmbeddedCodeLab

The takeaway of this coding project is demonstration of Assignments solved by me in C,  of Embedded Systems University of Texas at Austin under Prof. Valvano. I have restrictions uploading entire set of solved assignments.
I plan to extend this by uploading the Code of my current Embedded System project at PTW for Automation of Coolant Monitoring as Real Time Application Project of this learning under Mr.Krebs .
 Lab2_HelloLaunchPad (Hello World Exercise)
 Lab4_IO (Pin IO Configuration in C )
 Lab5_FunctionsInC (Light ON/OFF implementation)
 Lab6_BranchingFunctionsDelays (Light ON/OFF at Flicker Rate implementation )
 Lab7_HeartBlock (Heart Beat Implementation)
 Lab8_SwitchLEDinterface (Simple LED Circuit)


4. HeatTransferSolver

The takeaway of this C++ coding project is a Solver in Finite Difference Method for the solution of the 2D transient heat equation. The implementation is very relevant with application used by COMSOL for Heat Solver.
A bit complicated due to library functions of Eigen-Dense. Parallel Computing via OpenMP Pragma Directives would be ambitious.
This is solo project.
 
|--Finite Difference Methods Applied to the Heat Equation in 1D and 2D.pdf (I recommend reading this theory)
|--Programming_Tools_Project_Description.pdf (Theory)
|--main.cpp (Solver)
|--Makefile (This is how you run the code)

5. MachineLearningDataChallenge (Colab Project India-China-German-Egpyt)

This was my first ML Project. The takeaway of this project was Machine Learning Algorithm for Prediction of Quality of Piston-Cylinder Assembly  deployed for FlowFactory @ TU Darmstadt. One key feature was the BigData nature of problem (55GB exactly) due to requirement of Features for Algorithm to behave in Quality Critical Data . My role was Data Preparation and Feature Calculation , I used my knowledge of Python Numpy , Pandas and learnt scikit. Ahmad was Documenting the project while Wang(he got covid 3 times) and Tim( i see him in our UniFit Gym) performed Ensembled Random-Forest due to their expertise in PyTorch and Keras. I was active listener while they implemented their Algorithms and I learnt from my colleagues to apply the knowledge for Project " . MLOpsForQualtiyDetectionWithDashboard " covered next. GPU requirements could only be met by TU Darmstadts Lichtenberg Cluster. I post limited code to my part only.


|--20231120_PTW_Data Challenge Kick-Off (1).pdf (Theory for ref.)
|--data_preparation.py(Run this code for data preparation Python Numpy, Pandas ,Sckit used)
|--feature_calculation.py (calculate features)
|--model_gp_45.joblib (joblib data)
|--environment.yml (Environment Variables)


6. MLOpsForQualtiyDetectionWithDashboard (Colab Project India-Pakistan-German-Romania-Columbia)

This was colab project done in Tutorial ML for Production Application. Direct implementation of Predictive Maintainance with Machine Learning(MLOps).The output of this project is deployable Python Dashboard GUI integrated with Machine Learning Algorithm to Automatically Detect Quality of CNC Milling Tool during operation and it alerts the machine operator when Tool has to be replaced. 
I worked on Python PyTorch , Keras , Numpy , Pandas , Scikit(learnt from MachineLearningDataChallenge) to develop algorithm for Random-Forest Implementation. One key important point that I trained and tested my model on BigData Feature Engineering (50GB exactly).
My Colleagues Umar-Lukas worked on Convolution Neural Network implementation(very interesting) ,SVM Algorithm(Sven-David) . The Dashboard implementation incorporated SVM algorithm as it featured to be best performer followed by Random-Forest. This was implemented and test on CNC Milling Machine at FlowFactory.

|--Algorithms
|  |--seml_2025-main_withdashboard.zip(run code to gen dashboard)
|  |--seml_2025-feature-random_forest_Sritej.zip(run code to gen random forest algo)
|  |--seml_2025-data_PTWFactoryQuality.zip(Contains Data Set)
|  |--seml_2025-feature_david_sven.zip(run code to gen SVM algo)
|  |--seml_2025-feature_umar_lukas.zip(run code to gen CNN algo)
|--SEML_2025_documentation.pdf (Theory for ref.)

7. ScriptsForusefulForCFD

The takeaway of this Project Directory is it contains important codes required for Analysis of CFD Simulation. It takes 40% of research time in Post Processing Plots for correct strategic theoretical representation . I would like to share them. I am actively updating and uploading new codes in this area.

|--turbulenceTotal (run this code by pasting in src file OpenFOAM CFD Simulation)
|  |-- Make (run this file with correct Path)
|  |-- turbulenceFieldsTemplates.C..
|--extractPointsToh5.ipynb (run this code directly uses h5py , it converts given database into Results and h5 data)





