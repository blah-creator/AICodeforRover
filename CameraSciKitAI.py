import csv #MAKES FILES
import numpy as np #MATH THINGS
from pathlib import Path #HELPS WITH DIRECTORIES
from PIL import Image #OPEN AND MANIPULATE IMAGES
from sklearn.model_selection import train_test_split #TRAINS AND TESTS DATA
from sklearn.neighbors import KNeighborsClassifier #THE AI MODEL
from sklearn.metrics import accuracy_score #SHOWS HOW GOOD IT IS
import joblib #SAVES THE AI BRAIN TO A FILE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

CSV_PATHS = [Path("data/rover_data_camera.csv"),Path("data/live_feedback_rover.csv"),]#CSV FILE LOCATION
MODEL_PATH = Path("v2_ai_model.pkl") #LOCATION OF AI BRAIN(PKL is used to save complex models in python)

ACTIONS_TO_IDX = {"FWD":0,"BACK":1,"RIGHT":2,"LEFT":3} #LABELS WHAT THE ROVER CAN DO AND ASSIGNS A NUMBER FOR LATER

IMAGE_SIZE = 16 #IMAGE SIZE IS 16x16 PIXELS

def load_data():
    X = [] #THIS HOLDS NUMBERS FROM ACTIONS
    Y = [] #THIS HOLDS GOOD OR BAD
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"Warning {csv_path} does not exist. Skipping")
            continue

        print(f"Reading from {csv_path}")
        count = 0
        with csv_path.open("r") as f: #OPEN DATA FOR READING
            reader = csv.DictReader(f) #READ ROWS LIKE DICTIONARY (row [ACTION])
            for row in reader: #processes each line in the csv
                if row["label"].lower() != "good": #skip row if bad so it learns only good things
                    continue

                action = row["action"].upper() #CONVERTS EVERYTHING TO UPPERCASE so fwd = FWD
                if action not in ACTIONS_TO_IDX: #SKIPS OVER WEIRD ACTIONS
                    continue

                img_path = Path(row["image_path"]) #ASSIGNS VARIABLE TO THE ROW OF IMAGE PATH
                if not img_path.exists(): #MAKES SURE IT EXISTS
                    continue

                img = Image.open(img_path).convert("L") #CONVERTS TO GRAYSCALE SO IT IS EASIER FOR AI TO READ
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE)) #MAKES IMAGE 16x16 PIXELS

                arr = np.array(img).flatten() / 255.0 #TURNS IMAGE INTO ARRAY OF NUMBERS FROM 0 TO 1 THAT SHOWS THE BRIGHTNESS OF THE PIXEL IT DESCRIBES
                X.append(arr) #ADD THE ARRAY OF IMAGES TO X VALUES
                Y.append(ACTIONS_TO_IDX[action]) #ADD THE GOOD ACTION BECAUSE BAD WAS SKIPPED TO Y VALUES
                count+=1
        print(f"{count} from {csv_path}")
    X = np.array(X) #MAKE ARRAY OF X
    Y = np.array(Y) #MAKE ARRAY OF Y

    print("Loaded", len(X), "Samples")#SHOWS HOW MANY SAMPLES WERE TAKEN IN
    return X,Y #PUTS THESE AS THE VALUES OF X AND Y NOW

def main():
    X,Y = load_data() #MAKES LOCAL VARIABLES FOR X AND Y THAT WENT THROUGH LOAD DATA FUNCTION
    X_train, X_validate, Y_train, Y_validate = train_test_split(X,Y, test_size=0.2, random_state=42)
    #TAKES THE X AND Y DATA AND USES 80% OF IT TO TRAIN ITS MODEL AND 20% TO TEST IT THEN 42 SPLITS IT UP BUT DOESN'T KEEP IT RANDOM
    #model = KNeighborsClassifier(n_neighbors=5)
    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)#MAKES THE AI MODEL
    model.fit(X_train, Y_train) #TRAINS THE AI MODEL TO FIT THE DATA AND MAKE CORRECT GUESSES
    prediction = model.predict(X_validate) #IT PREDICTS THE 20% LAID OUT FOR TESTING WITH A Y_VALUE
    acc = accuracy_score(Y_validate, prediction)#CHECKS THE Y_VALIDATE WITH THE PREDICTIONS AND GIVES IT THE SCORE
    print(classification_report(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print("Accuracy: ", acc) #SHOWS THE ACCURACY

    joblib.dump(model, MODEL_PATH) #SAVES THE AI INTO A FILE SO WE CAN USE IT LATER (AI MODEL VARIABLE, PLACE IT GOES)
    print("MODEL SAVED TO: ", MODEL_PATH)

if __name__ == "__main__":
    main()