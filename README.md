# Weather application based on FMI's public observation data


## Instructions for usage of the application


1. Download .zip file by clicking on the green "Code" button on this page and extract the downloaded file, or clone this repository with `git clone`.

2. Start the Flask server by running the file `app.py` that exists in the map `/backend`.

4. When the server is running, open a new terminal.
    *(In Visual Studio Code this can be done with either clicking "Terminal" in the top left corner and choosing "New Terminal", 
    or by clicking the plus icon in the terminal tab below the code field)*

5. Then run the Qt client with the command: 
    `python client/qt.py`
    to open the graphical interface.

6. A new window will open where you can fetch actual weather data and show temperature predicitions with the result in a graph.

7. If you want to train the model again you can run `train_model.py`, that is located in the `/utils` folder.

## Results

<img width="1199" height="630" alt="image" src="https://github.com/user-attachments/assets/ce0de283-dd0d-4c5e-aeeb-28550175660c" />
