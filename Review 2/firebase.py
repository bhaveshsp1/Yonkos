import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-attendance-5b38b-default-rtdb.firebaseio.com/'
})
attendance=1
# Function to update data in Realtime Database
def update_data(path, updates):
    ref = db.reference(path)
    ref.update(updates)

# Example updates to be applied
roll="22BIT0226"
updates = {
    roll:1
}


# Call the function to update data in Realtime Database
update_data("/users", updates)



updates = {
    "id":"STUDENT",
    "status":attendance,
    "time":formatted_datetime,
    "uid":roll
}
update_data("/attendence/"+roll, updates)