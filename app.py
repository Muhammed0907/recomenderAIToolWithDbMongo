from flask import Flask, request,redirect, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient
from flasgger import Swagger, swag_from
import threading
import numpy as np
import pandas as pd
import faiss
from model.wpost import WPPosts,db,GetToolsInfo
from utils.tool import RowToTextualRepresentation
from utils.train import TrainModel,training_status,Request2Ollama,AddNewAIToolInModel
import random
import toml
import argparse

parser = argparse.ArgumentParser(description="config: dev or prod")

parser.add_argument("--config", type=str, help="Get",default="dev",required=False)

args = parser.parse_args()

if args.config == "prod":
    CONFIG_FILE = './config/config_dev.toml'
else:
    CONFIG_FILE = './config/config.toml'

config = toml.load(CONFIG_FILE)

app = Flask(__name__)
swagger = Swagger(app)


MYSQL_CONF = config["mysql"]
USERNAME = MYSQL_CONF["username"]
PASSWORD = MYSQL_CONF["password"]
HOST = MYSQL_CONF["host"]
PORT = MYSQL_CONF["port"]
DB = MYSQL_CONF["db"]

MYSQL_URL = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'

app.config['SQLALCHEMY_DATABASE_URI'] = MYSQL_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app=app)


RETURN_TOP = 5

def getUsersFromMongo():
    MONGO_CONF = config["mongodb"]
    MONGO_URL = f'mongodb://{MONGO_CONF["username"]}:{MONGO_CONF["password"]}@{MONGO_CONF["host"]}:{MONGO_CONF["port"]}/'
    cl = MongoClient(MONGO_URL)
    db = cl["my_database"]
    users = db["users"]
    return users



# REGISTER USER-------------------
@app.route("/register", methods=["POST"])
@swag_from("swagger_docs/register.yml")
def registerUser():
    data = request.get_json()
    user_id = data.get("user_id")
    user_name = data.get("user_name")

    if not user_id or not user_name:
        return jsonify({"error": "User ID and user name are required"}), 400

    users = getUsersFromMongo()

    existing_user = users.find_one({"user_id": user_id})
    if existing_user:
        return jsonify({"error": "User already exists"}), 409

    new_user = {
        "user_id": user_id,
        "user_name": user_name,
        "searched_history": [],
        "click_history": []
    }
    users.insert_one(new_user)

    return jsonify({"message": "Registration successful"}), 201


# LOGIN USER-------------------
@app.route("/login", methods=["POST"])
@swag_from("swagger_docs/login.yml")
def loginUser():
    data = request.get_json()
    user_id = data.get("user_id")
    user_name = data.get("user_name")

    if not user_id or not user_name:
        return jsonify({"error": "User ID and user name are required"}), 400

    users = getUsersFromMongo()
    
    user = users.find_one({"user_id": user_id, "user_name": user_name})
    if user:
        response = make_response(jsonify({"message": "Login successful"}))
        response.set_cookie("user_id", user_id)

        return response
    else:
        return jsonify({"error": "User not found"}), 404



def GetUSERInfo(userId:str):
    users = getUsersFromMongo()

    user = users.find_one({"user_id": userId})
    if user:
        user_info = {
            "user_id": user.get("user_id"),
            "user_name": user.get("user_name"),
            "searched_history": user.get("searched_history", []),
            "click_history": user.get("click_history", [])
        }
    else:
        return None
    
    return user_info


# GET  USER PROFILE-------------------
@app.route("/userprofile", methods=["GET"])
@swag_from("swagger_docs/userprofile.yml")
def getUserProfile():
    user_id = request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    userInfo = GetUSERInfo(userId=user_id)
    if userInfo:
        return jsonify(userInfo), 200
    
    return jsonify({"error": "User not found"}), 404


# COLLECT USER DATA(HISTORY)-------------------
@app.route("/collect_userinfo", methods=["POST"])
@swag_from("swagger_docs/collect_userinfo.yml")
def collect_userinfo():
    user_id = request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    selected = request.args.get("select")
    if not selected:
        return jsonify({"error": "No  selected provided."}), 400

    data = request.json
    historyInfo = data.get("info")
    if not historyInfo:
        return jsonify({"error": "No info provided."}), 400
    
    users = getUsersFromMongo()
    user = users.find_one({"user_id": user_id})
    if user:
        if selected == "search":
            UserSearchedHis = user.get("searched_history", [])
            UserSearchedHis = UserSearchedHis[len(UserSearchedHis)-2:]
            UserSearchedHis.append(historyInfo)
            result = users.update_one(
                {"user_id": user_id},
                {"$set": {"searched_history": UserSearchedHis}}
            )
        elif selected == "clicked_tool":
            UserClickHistory = user.get("click_history", [])
            UserClickHistory = UserClickHistory[len(UserClickHistory)-2:]
            UserClickHistory.append(historyInfo)
            result = users.update_one(
                {"user_id": user_id},
                {"$set": {"click_history": UserClickHistory}}
            )
        else:
            return jsonify({"error": "select query param invalid, must: search,clicked_tool"}), 404

        if result.modified_count == 1:
            return jsonify({"message": f"{selected} history updated successfully"}), 200
        else:
            return jsonify({"error": "mongo not updated or already exist"}), 404

    return jsonify({"error": "User not found"}), 404
        


# GET AI TOOLS-------------------
@app.route("/recomen_tool",methods=["GET"])
@swag_from("swagger_docs/recomend.yml")
def recomen_tool():
    user_id = request.cookies.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    userInfo = GetUSERInfo(user_id)
    if userInfo == None:
        return jsonify({"error": "User not found"}), 404 
    
    userHistoryInfo = userInfo['searched_history'][::-1] + userInfo['click_history'][::-1]
    random.shuffle(userHistoryInfo)

    trainedAitoolEmbedModel = faiss.read_index(config["model"]["name"])

    historyTexualRep = ', '.join(map(str,userHistoryInfo))

    res = Request2Ollama(historyTexualRep)

    embed = np.array([res.json()["embedding"]],dtype="float32")
    _,I = trainedAitoolEmbedModel.search(embed,RETURN_TOP)

    df = pd.read_csv("results.csv")
    results = df.iloc[I.flatten()]
    results_json = results.to_dict(orient='records')

    return jsonify(results_json), 200


# TRAIN AVAILABLE TOOLS-------------------
@app.route("/train", methods=["GET"])
@swag_from("swagger_docs/train_model.yml")
def getData():
    toolsInfo = GetToolsInfo()
    textualRep = RowToTextualRepresentation(toolsInfo)

    threading.Thread(target=TrainModel, args=(textualRep,config["model"]["name"])).start()

    return "Training started", 200


# ADD NEW TOOL IN TRAINED MODEL
@app.route("/add_new_data", methods=["POST"])
@swag_from("swagger_docs/add_new_data.yml")
def addNewData():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid data format. Expected a list of tools with descriptions."}), 400

    try:
        textualData = [f"Tool: {item['tool']}\nDescription: {item['description']}\n" for item in data]

    except KeyError:
        return jsonify({"error": "Each item must include 'tool' and 'description' fields."}), 400

    modelPath = config["model"]["name"]  
    print(modelPath)
    try:
        AddNewAIToolInModel(modelPath, textualData)
        csv_data = [{'post_title': item['tool'], 'post_content': item['description']} for item in data]
        new_df = pd.DataFrame(csv_data)
        # new_df['/index'] = range(savedInd, savedInd + len(new_df))
       
        csv_file_path = 'results.csv'
        new_df.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False, encoding='utf-8')

        return jsonify({"message": "New data added successfully to the model."}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to add new data to model: {str(e)}"}), 500


# CHECK TRAINING PROCCESS STATUS-------------------
@app.route("/result", methods=["GET"])
@swag_from("swagger_docs/result.yml")
def check_result():
    
    if training_status["is_training"]:
        return jsonify({
            "status": "Training in progress",
            "current_instance": training_status["current_instance"],
            "total_instances": training_status["total_instances"]
        }), 200

    elif training_status["success"]:
        return jsonify({"status": "Training complete"}), 200

    else:
        return jsonify({"status": "Training not started"}), 200


@app.route("/")
def main():
    return redirect("http://localhost:5151/apidocs", code=302)

app.run(host="localhost", port=5151, debug=True)
